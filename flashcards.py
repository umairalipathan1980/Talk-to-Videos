# flashcards.py

import json
import random
from typing import List, Dict, Any

class FlashcardGenerator:
    """Class to generate flashcards from video content using the RAG system."""
    
    def __init__(self):
        """Initialize the flashcard generator."""
        pass
    
    def generate_flashcards(self, rag_system, api_key, transcript=None, num_cards=10, model="gpt-4o") -> List[Dict[str, str]]:
        """
        Generate flashcards based on the video content.
        
        Args:
            rag_system: The initialized RAG system with video content
            api_key: OpenAI API key
            transcript: The full transcript text (optional)
            num_cards: Number of flashcards to generate (default: 10)
            model: The OpenAI model to use
            
        Returns:
            List of flashcard dictionaries with 'front' and 'back' keys
        """
        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI
        
        # Initialize language model
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model=model,
            temperature=0.4
        )
        
        # Create the prompt for flashcard generation
        prompt = f"""
        Create {num_cards} educational flashcards based on the video content.
        
        Each flashcard should have:
        1. A front side with a question, term, or concept
        2. A back side with the answer, definition, or explanation
        
        Focus on the most important and educational content from the video. 
        Create a mix of different types of flashcards:
        - Key term definitions
        - Conceptual questions
        - Fill-in-the-blank statements
        - True/False questions with explanations
        
        Format your response as a JSON array of objects with 'front' and 'back' properties.
        Example:
        [
            {{"front": "What is photosynthesis?", "back": "The process by which plants convert light energy into chemical energy."}},
            {{"front": "The three branches of government are: Executive, Legislative, and _____", "back": "Judicial"}}
        ]
        
        Make sure your output is valid JSON format with exactly {num_cards} flashcards.
        """
        
        try:
            # Determine the context to use
            if transcript:
                # Use the full transcript if provided
                # Create messages for the language model
                messages = [
                    {"role": "system", "content": f"You are an educational content creator specializing in creating effective flashcards. Use the following transcript from a video to create educational flashcards:\n\n{transcript}"},
                    {"role": "user", "content": prompt}
                ]
            else:
                # Fallback to RAG system if no transcript is provided
                relevant_docs = rag_system.vector_store.similarity_search(
                    "key points and educational concepts in the video", k=15
                )
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create messages for the language model
                messages = [
                    {"role": "system", "content": f"You are an educational content creator specializing in creating effective flashcards. Use the following context from a video to create educational flashcards:\n\n{context}"},
                    {"role": "user", "content": prompt}
                ]
            
            # Generate flashcards
            response = llm.invoke(messages)
            content = response.content
            
            # Extract JSON content in case there's text around it
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                flashcards = json.loads(json_content)
            else:
                # Fallback in case of improper JSON formatting
                raise ValueError("Failed to extract valid JSON from response")
            
            # Verify we have the expected number of cards (or adjust as needed)
            actual_cards = min(len(flashcards), num_cards)
            flashcards = flashcards[:actual_cards]
            
            # Validate each flashcard has required fields
            validated_cards = []
            for card in flashcards:
                if 'front' in card and 'back' in card:
                    validated_cards.append({
                        'front': card['front'],
                        'back': card['back']
                    })
            
            return validated_cards
        
        except Exception as e:
            # Handle errors gracefully
            print(f"Error generating flashcards: {str(e)}")
            # Return a few basic flashcards in case of error
            return [
                {"front": "Error generating flashcards", "back": f"Please try again. Error: {str(e)}"},
                {"front": "Tip", "back": "Try regenerating flashcards or using a different video"}
            ]
    
    def shuffle_flashcards(self, flashcards: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Shuffle the order of flashcards"""
        shuffled = flashcards.copy()
        random.shuffle(shuffled)
        return shuffled

