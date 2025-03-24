from langchain_openai import ChatOpenAI
import streamlit as st
import random  # Add this import at the top


class QuizGenerator:
    def __init__(self):
        pass
    
    def generate_quiz(self, rag_system, api_key, transcript=None, model="gpt-4o", temperature=0.4):
        """
        Generate quiz questions based on the video transcript
        
        Args:
            rag_system: The RAG system with vector store2
            api_key: OpenAI API key
            transcript: The full transcript text (optional)
            model: Model to use for question generation
            temperature: Creativity level (0.0-1.0)
            
        Returns:
            list: List of question objects
        """
        if not rag_system:
            st.error("Please transcribe the video first before creating a quiz!")
            return []
        
        # Create a temporary LLM with slightly higher temperature for more creative questions
        creative_llm = ChatOpenAI(
            openai_api_key=api_key,
            model=model,
            temperature=temperature
        )

        num_questions = 10
        
        # Prompt to generate quiz
        prompt = f"""Based on the video transcript, generate {num_questions} multiple-choice questions to test understanding of the content.
        For each question:
        1. The question should be specific to information mentioned in the video
        2. Include 4 options (A, B, C, D)
        3. Clearly indicate the correct answer
        
        Format your response exactly as follows for each question:
        QUESTION: [question text]
        A: [option A]
        B: [option B]
        C: [option C]
        D: [option D]
        CORRECT: [letter of correct answer]
       
        Make sure all questions are based on facts from the video."""
        
        try:
            if transcript:
                # If we have the full transcript, use it
                messages = [
                    {"role": "system", "content": f"You are given the following transcript from a video:\n\n{transcript}\n\nUse this transcript to create quiz questions according to the instructions."},
                    {"role": "user", "content": prompt}
                ]
                
                response = creative_llm.invoke(messages)
                response_text = response.content
            else:
                # Fallback to RAG approach if no transcript is provided
                relevant_docs = rag_system.vector_store.similarity_search(
                    "what are the main topics covered in this video?", k=5
                )
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Use the creative LLM with context to generate questions
                messages = [
                    {"role": "system", "content": f"You are given the following context from a video transcript:\n\n{context}\n\nUse this context to create quiz questions according to the instructions."},
                    {"role": "user", "content": prompt}
                ]
                
                response = creative_llm.invoke(messages)
                response_text = response.content
        except Exception as e:
            # Fallback to the regular RAG system if there's an error
            st.warning(f"Using standard question generation due to error: {str(e)}")
            response_text = rag_system.ask(prompt)
        
        return self.parse_quiz_response(response_text)

    # The rest of the class remains unchanged
    def parse_quiz_response(self, response_text):
        """
        Parse the LLM response to extract questions, options, and correct answers
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            list: List of parsed question objects
        """
        quiz_questions = []
        current_question = {}
        
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('QUESTION:'):
                if current_question and 'question' in current_question and 'options' in current_question and 'correct' in current_question:
                    quiz_questions.append(current_question)
                current_question = {
                    'question': line[len('QUESTION:'):].strip(),
                    'options': [],
                    'correct': None
                }
            elif line.startswith(('A:', 'B:', 'C:', 'D:')):
                option_letter = line[0]
                option_text = line[2:].strip()
                current_question.setdefault('options', []).append((option_letter, option_text))
            elif line.startswith('CORRECT:'):
                current_question['correct'] = line[len('CORRECT:'):].strip()
        
        # Add the last question
        if current_question and 'question' in current_question and 'options' in current_question and 'correct' in current_question:
            quiz_questions.append(current_question)
        
        # Randomize options for each question
        randomized_questions = []
        for q in quiz_questions:
            # Get the original correct answer
            correct_letter = q['correct']
            correct_option = None
            
            # Find the correct option text
            for letter, text in q['options']:
                if letter == correct_letter:
                    correct_option = text
                    break
            
            if correct_option is None:
                # If we can't find the correct answer, keep the question as is
                randomized_questions.append(q)
                continue
                
            # Create a list of options texts and shuffle them
            option_texts = [text for _, text in q['options']]
            
            # Create a copy of the original letters
            option_letters = [letter for letter, _ in q['options']]
            
            # Create a list of (letter, text) pairs
            options_pairs = list(zip(option_letters, option_texts))
            
            # Shuffle the pairs
            random.shuffle(options_pairs)
            
            # Find the new position of the correct answer
            new_correct_letter = None
            for letter, text in options_pairs:
                if text == correct_option:
                    new_correct_letter = letter
                    break
            
            # Create a new question with randomized options
            new_q = {
                'question': q['question'],
                'options': options_pairs,
                'correct': new_correct_letter
            }
            
            randomized_questions.append(new_q)
        
        return randomized_questions
    
    def calculate_quiz_results(self, questions, user_answers):
        """
        Calculate quiz results based on user answers
        
        Args:
            questions: List of question objects
            user_answers: Dictionary of user answers keyed by question_key
            
        Returns:
            tuple: (results dict, correct count)
        """
        correct_count = 0
        results = {}
        
        for i, question in enumerate(questions):
            question_key = f"quiz_q_{i}"
            user_answer = user_answers.get(question_key)
            correct_answer = question['correct']
            
            # Only count as correct if user selected an answer and it matches
            is_correct = user_answer is not None and user_answer == correct_answer
            if is_correct:
                correct_count += 1
            
            results[question_key] = {
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            }
        
        return results, correct_count
