import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate

class VideoRAG:
    def __init__(self, api_key=None, chunk_size=1000, chunk_overlap=200):
        """Initialize the RAG system with OpenAI API key."""
        # Use provided API key or try to get from environment
        self.api_key = api_key if api_key else st.secrets["OPENAI_API_KEY"]
        if not self.api_key:
            raise ValueError("OpenAI API key is required either as parameter or environment variable")
            
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model="gpt-4o",
            temperature=0
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def create_vector_store(self, transcript):
        """Create a vector store from the transcript."""
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(transcript)
        
        # Create vector store
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        
        # Create prompt template for the RAG system
        system_template = """You are a specialized AI assistant that answers questions about a specific video. 
        
        You have access to snippets from the video transcript, and your role is to provide accurate information ONLY based on these snippets.
        
        Guidelines:
        1. Only answer questions based on the information provided in the context from the video transcript, otherwise say that "I don't know. The video doesn't cover that information."
        2. The question may ask you to summarize the video or tell what the video is about. In that case, present a summary of the context. 
        3. Don't make up information or use knowledge from outside the provided context
        4. Keep your answers concise and directly related to the question
        5. If asked about your capabilities or identity, explain that you're an AI assistant that specializes in answering questions about this specific video
        
        Context from the video transcript:
        {context}
        
        Chat History:
        {chat_history}
        """
        
        user_template = "{question}"
        
        # Create the messages for the chat prompt
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
        ]
        
        # Create the chat prompt
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Initialize the RAG chain with the custom prompt
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )
        
        return len(chunks)
    
    def set_chat_history(self, chat_history):
        """Set chat history from external session state."""
        if not self.memory:
            return
            
        # Clear existing memory
        self.memory.clear()
        
        # Convert standard chat history format to LangChain message format
        for message in chat_history:
            if message["role"] == "user":
                self.memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                self.memory.chat_memory.add_ai_message(message["content"])
    
    def ask(self, question, chat_history=None):
        """Ask a question to the RAG system."""
        if not self.chain:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        # If chat history is provided, update the memory
        if chat_history:
            self.set_chat_history(chat_history)
        
        # Get response
        response = self.chain.invoke({"question": question})
        return response["answer"]
