import streamlit as st
import os
from pathlib import Path
import time
import base64
import random
import streamlit.components.v1 as components
import markdown
import subprocess
import time
from transcriber import (
    process_video, process_video_download, process_video_transcribe,
    get_video_info, verify_ffmpeg, transcript_exists, read_transcript
)
from rag_system import VideoRAG
from dotenv import load_dotenv
from quiz import QuizGenerator
from flashcards import FlashcardGenerator
from summary import SummaryGenerator

# New imports for speech functionality
import numpy as np
import tempfile
import sounddevice as sd
import wave

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Talk to Videos",
    page_icon="🎬",
    layout="wide"
)

# Set output directory as a constant
OUTPUT_DIR = "output"
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a directory for audio files if it doesn't exist
audio_dir = os.path.join(tempfile.gettempdir(), "streamlit_audio")
os.makedirs(audio_dir, exist_ok=True)

# Function to clear all session state and restart the app
def clear_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
# Function to get OpenAI API key from environment
def get_api_key():
    return st.secrets["OPENAI_API_KEY"]

##Function to convert text to speech. 
def text_to_speech(text, api_key, voice="nova"):
    output_filename = None
    try:
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",  
            voice=voice,
            input=text
        )
        
        # Save the audio to a file in our audio directory
        output_filename = os.path.join(audio_dir, f"response_{int(time.time())}.mp3")
        response.stream_to_file(output_filename)
        
        # Read the saved audio file
        with open(output_filename, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Delete the file after reading its contents into memory
        if os.path.exists(output_filename):
            os.remove(output_filename)
        
        return audio_bytes, None  # Return None instead of filename since it's deleted
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None, None
    finally:
        # Extra safeguard to ensure file is deleted even if something unexpected happens
        if output_filename and os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except Exception as e:
                print(f"Error deleting speech file in finally block: {e}")

# Helper function to auto-play audio
def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# Initialize session state variables only if they don't exist
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.rag_initialized = False
    st.session_state.rag_system = None
    st.session_state.processing = False
    st.session_state.video_title = ""
    st.session_state.chat_history = []
    st.session_state.transcript = None
    st.session_state.auto_setup_done = False
    st.session_state.video_info = None
    st.session_state.youtube_url = ""
    st.session_state.ffmpeg_verified = False
    # Quiz-related session state
    st.session_state.quiz = []  
    st.session_state.quiz_questions = []
    st.session_state.show_quiz = False
    st.session_state.quiz_submitted = False
    st.session_state.user_answers = {}
    st.session_state.quiz_results = {}
    st.session_state.current_quiz_question = 0 
    st.session_state.quiz_completed = False  
    # Initialize quiz generator
    st.session_state.quiz_generator = QuizGenerator()
    # Summary-related session state
    st.session_state.hierarchical_summary = ""
    st.session_state.show_summary_popup = False
    st.session_state.summary_generator = SummaryGenerator()
    # Flashcard-related session state
    st.session_state.flashcards = []
    st.session_state.show_flashcards = False
    st.session_state.current_flashcard_index = 0
    st.session_state.show_flashcard_answer = False
    st.session_state.flashcard_generator = FlashcardGenerator()
    st.session_state.transcription_model = "gpt-4o-transcribe"
    # Speech-related session state
    st.session_state.enable_speech = True
    st.session_state.selected_voice = "nova"
    st.session_state.audio_file_path = None

# Check if we need to rerun due to question submission
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False
if st.session_state.need_rerun:
    st.session_state.need_rerun = False
    st.rerun()

##code to check whether we need to generate speech
if 'generate_speech_for' not in st.session_state:
    st.session_state.generate_speech_for = None

#### Check if we need to generate speech for a response
if st.session_state.enable_speech and st.session_state.generate_speech_for:
    api_key = get_api_key()
    if api_key:
        with st.spinner("Generating speech..."):
            audio_bytes, _ = text_to_speech(
                st.session_state.generate_speech_for, 
                api_key, 
                voice=st.session_state.selected_voice
            )
            if audio_bytes:
                autoplay_audio(audio_bytes)
    
    # Clear the flag after the speech has been generated
    st.session_state.generate_speech_for = None
##############################

# Function to record audio
def record_audio(duration=5, sample_rate=44100):
    """Record audio for a specified duration with a visual countdown."""
    # Create a placeholder for recording status
    status_placeholder = st.empty()
    
    # Record audio
    status_placeholder.write(f"Listening... {duration} seconds remaining")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    
    # Display countdown
    for remaining in range(duration-1, 0, -1):
        time.sleep(1)
        status_placeholder.write(f"Listening... {remaining} seconds remaining")
    
    sd.wait()
    status_placeholder.write("Voice input complete!")
    return audio

def save_audio(audio, filename, sample_rate=44100):
    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(1)
    wavfile.setsampwidth(2)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes((audio * 32767).astype(np.int16).tobytes())
    wavfile.close()
    return filename

def transcribe_audio(audio_file, api_key):
    try:
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        with open(audio_file, 'rb') as audio:
            response = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",  
                file=audio
            )
        return response.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Function to check if a file exists at a given path
def is_local_file(path):
    """Check if a path points to an existing local file"""
    return os.path.isfile(path) and os.path.exists(path)

# Function to generate flashcards
def generate_flashcards():
    if not st.session_state.rag_initialized:
        st.error("Please transcribe the video first before creating flashcards!")
        return
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return
    
    # Get the transcript from session state
    transcript = st.session_state.transcript if 'transcript' in st.session_state else None
    
    # Use the flashcard generator to create flashcards
    with st.spinner("Generating flashcards..."):
        try:
            flashcards = st.session_state.flashcard_generator.generate_flashcards(
                st.session_state.rag_system, 
                api_key,
                transcript=transcript,  # Pass the transcript to the generator
                num_cards=10  # Default number of cards to generate
            )
            
            # Check if we got valid flashcards
            if not flashcards or len(flashcards) == 0:
                st.error("No flashcards were generated. Please try again.")
                return
                
            # Store the flashcards in session state
            st.session_state.flashcards = flashcards
            st.session_state.show_flashcards = True
            st.session_state.show_quiz = False  # Hide quiz when showing flashcards
            st.session_state.current_flashcard_index = 0
            st.session_state.show_flashcard_answer = False
            
            # Reset quiz state
            st.session_state.current_quiz_question = 0
            st.session_state.quiz_results = []
            st.session_state.quiz_completed = False
        
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")
            return

# Function to show next flashcard
def next_flashcard():
    if st.session_state.flashcards:
        total_cards = len(st.session_state.flashcards)
        st.session_state.current_flashcard_index = (st.session_state.current_flashcard_index + 1) % total_cards
        st.session_state.show_flashcard_answer = False

# Function to show previous flashcard
def prev_flashcard():
    if st.session_state.flashcards:
        total_cards = len(st.session_state.flashcards)
        st.session_state.current_flashcard_index = (st.session_state.current_flashcard_index - 1) % total_cards
        st.session_state.show_flashcard_answer = False

# Function to toggle showing the answer
def toggle_answer():
    st.session_state.show_flashcard_answer = not st.session_state.show_flashcard_answer

# Function to create a download link
def get_download_link(file_path, link_text):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

# Function to validate YouTube URL
def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url

# Function to initialize RAG system
def init_rag_system(transcript, video_title):
    # Get API key from environment
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
        
    # Create RAG system with constant chunk values
    rag = VideoRAG(api_key=api_key, chunk_size=800, chunk_overlap=200)
    chunk_count = rag.create_vector_store(transcript)
    st.session_state.rag_system = rag
    st.session_state.rag_initialized = True
    
    # Clear any previous chat history
    st.session_state.chat_history = []
    
    # Add system welcome message
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": f"I'm ready to answer questions about '{video_title}'. What would you like to know?"
    })
    
    return chunk_count

# Function to display video details
def display_video_info(info):
    video_title = info.get('title', 'Unknown Title')
    video_duration = info.get('duration', 0)
    thumbnail_url = info.get('thumbnail', '')
    
    st.subheader("Video Preview")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if thumbnail_url:
            st.image(thumbnail_url, use_container_width=True)
    with col_b:
        st.subheader(video_title)
        st.write(f"Duration: {int(video_duration//60)}:{int(video_duration%60):02d}")
        st.write(f"Channel: {info.get('uploader', 'Unknown')}")
        
        # Check if transcript already exists
        youtube_url = st.session_state.youtube_url if 'youtube_url' in st.session_state else ""
        if youtube_url and transcript_exists(youtube_url, OUTPUT_DIR):
            st.success("✅ Transcript already exists for this video")
            
            # Download links
            transcript_path = os.path.join(OUTPUT_DIR, f"{info.get('id', 'video')}_transcript.txt")
            if os.path.exists(transcript_path):
                transcript_link = get_download_link(transcript_path, "Download Existing Transcript")
                st.markdown(transcript_link, unsafe_allow_html=True)

### Function to handle question submission
def handle_question_submission():
    question = st.session_state.question_input
    
    if not question or not st.session_state.rag_initialized:
        return
        
    # Add user question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    try:
        # Get answer with spinner showing "Thinking..."
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_system.ask(question)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Set flag to generate speech on next run
        if st.session_state.enable_speech:
            st.session_state.generate_speech_for = answer
        
        # Clear the input field
        st.session_state.question_input = ""
        
        # Set a flag for next render cycle
        st.session_state.need_rerun = True
        st.rerun()
        
    except Exception as e:
        # Handle any errors
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        st.session_state.question_input = ""

### Function to handle speech input
def handle_speech_input():
    if not st.session_state.rag_initialized:
        st.error("Please transcribe the video first before using speech!")
        return

    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return
    
    ### Record audio
    audio_data = record_audio(duration=5)
    
    ### Save audio to a file
    audio_filename = os.path.join(audio_dir, f"input_{int(time.time())}.wav")
    save_audio(audio_data, audio_filename)
    
    ###Display audio playback
    # st.audio(audio_filename)
    
    # Transcribe audio with appropriate spinner
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_filename, api_key)
    
    if transcription:
        # Set the transcription as the question input
        st.session_state.question_input = transcription
        
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": transcription})
        
        try:
            # Get answer with "Thinking..." spinner
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_system.ask(transcription)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

             # Set flag to generate speech on next run
            if st.session_state.enable_speech:
                st.session_state.generate_speech_for = answer
            
            # Clear the input field
            st.session_state.question_input = ""

            if os.path.exists(audio_filename):
                try:
                    os.remove(audio_filename)
                except Exception as e:
                    print(f"Error deleting audio file: {e}")

            # Set a flag for next render cycle
            st.session_state.need_rerun = True
            st.rerun()
        except Exception as e:
            # Handle any errors
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.session_state.question_input = ""
            st.session_state.need_rerun = True
    else:
        st.error("Failed to transcribe speech. Please try again.")

# Function to handle URL or file change
def on_url_change():
    # Reset related states when URL changes
    if st.session_state.youtube_url != st.session_state.prev_url:
        st.session_state.prev_url = st.session_state.youtube_url
        st.session_state.video_info = None
        st.session_state.auto_setup_done = False
        st.session_state.rag_initialized = False
        # Reset quiz state
        st.session_state.quiz_questions = []
        st.session_state.show_quiz = False
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.quiz_results = {}
        # Reset summary state
        st.session_state.hierarchical_summary = ""
        st.session_state.show_summary_popup = False

# Function to handle file upload changes
def on_file_change():
    if "uploaded_video" in st.session_state and st.session_state.uploaded_video is not None:
        # Clear YouTube URL if a file is uploaded
        st.session_state.youtube_url = ""
        # Reset all other states
        st.session_state.video_info = None
        st.session_state.auto_setup_done = False
        st.session_state.rag_initialized = False
        # Reset quiz state
        st.session_state.quiz_questions = []
        st.session_state.show_quiz = False
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.quiz_results = {}
        # Reset summary state
        st.session_state.hierarchical_summary = ""
        st.session_state.show_summary_popup = False

# Function to handle transcribe button click
def on_transcribe_click():
    st.session_state.should_transcribe = True

# Function to create a new quiz (force new generation)
def create_new_quiz(transcript):
    if not st.session_state.rag_initialized:
        st.error("Please transcribe the video first before creating a quiz!")
        return
    
    # Clear existing quiz questions to force new generation
    st.session_state.quiz_questions = []
    generate_quiz()

# Function to generate quiz questions
def generate_quiz():
    if not st.session_state.rag_initialized:
        st.error("Please transcribe the video first before creating a quiz!")
        return
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return
    
    # Get the transcript from session state
    transcript = st.session_state.transcript if 'transcript' in st.session_state else None
    
    # Generate the quiz using QuizGenerator
    with st.spinner("Generating quiz questions..."):
        quiz = st.session_state.quiz_generator.generate_quiz(
            st.session_state.rag_system, 
            api_key,
            transcript=transcript  # Pass the transcript to the generate_quiz function
        )
    
    # Store the quiz in session state
    st.session_state.quiz = quiz
    st.session_state.show_quiz = True
    st.session_state.show_flashcards = False  # Hide flashcards when showing quiz
    st.session_state.current_quiz_question = 0
    st.session_state.quiz_results = []
    st.session_state.quiz_completed = False
    
    # Reset flashcard state
    st.session_state.current_flashcard_index = 0
    st.session_state.show_flashcard_answer = False

# Function to handle quiz submission
def submit_quiz():
    # Collect form data from session state
    user_answers = {}
    for i in range(len(st.session_state.quiz_questions)):
        question_key = f"quiz_q_{i}"
        if question_key in st.session_state:
            user_answers[question_key] = st.session_state[question_key]
        else:
            user_answers[question_key] = None  # No answer selected
    
    # Store user answers in session state
    st.session_state.user_answers = user_answers
    
    # Calculate results using the quiz generator
    results, correct_count = st.session_state.quiz_generator.calculate_quiz_results(
        st.session_state.quiz_questions,
        user_answers
    )
    
    st.session_state.quiz_results = results
    st.session_state.quiz_score = correct_count
    st.session_state.quiz_submitted = True

# Function to show summary popup
def show_summary_popup():
    """Show a pop-up with the hierarchical summary"""
    
    if not st.session_state.rag_initialized:
        st.error("Please transcribe the video first!")
        return
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return
    
    # Generate summary if not already generated
    if 'hierarchical_summary' not in st.session_state or not st.session_state.hierarchical_summary:
        summary = st.session_state.summary_generator.generate_summary(
            st.session_state.rag_system, 
            api_key
        )
        st.session_state.hierarchical_summary = summary
    else:
        summary = st.session_state.hierarchical_summary
    
    # Show the summary in a popup
    st.session_state.show_summary_popup = True

# App title and description
st.markdown("<h1><span style='vertical-align: middle'>🎬 Talk to Videos</span></h1>", unsafe_allow_html=True)

st.markdown("""
Explore educational videos through AI-powered conversations, quizzes, and flashcards. This tool transcribes video content and automatically develops a retrieval augment generation (RAG), 
allowing you to ask questions about the video and test your understanding with auto-generated quizzes and flashcards.
Perfect for learning, research, and content analysis.
""")

# Verify FFmpeg only once
if not st.session_state.ffmpeg_verified:
    try:
        ffmpeg_path, ffprobe_path = verify_ffmpeg()
        st.session_state.ffmpeg_verified = True
    except Exception as e:
        st.sidebar.error(f"❌ FFmpeg error: {str(e)}")
        st.stop()

# Sidebar settings
with st.sidebar:
    # ## Option to keep audio file
    # keep_audio = st.checkbox("Keep Audio File", 
    #                         value=True, 
    #                         help="If unchecked, the audio file will be deleted after transcription")
    
    # Add model selection
    transcription_model = st.selectbox(
        "Transcription Model",
        options=["gpt-4o-transcribe", "whisper-1"],
        index=0,
        help="Select the model to use for transcription"
    )
    # Display message for model limitations
    if transcription_model == "gpt-4o-transcribe":
        st.info("This model supports maximum file size of 50MB and video duration of 25 minutes (1500 seconds). For longer files, audio will be split into chunks automatically.")
    else:
        st.info("The whisper-1 model supports maximum file size of 25MB. For larger files, audio will be split into chunks automatically.")

    # Store the model in session state
    st.session_state.transcription_model = transcription_model
    
    # Add speech toggle
    st.markdown("---")
    st.subheader("Speech Settings")
    enable_speech = st.checkbox("Enable Speech", value=st.session_state.enable_speech, 
                               help="Enable speech input and output for interacting with the video")
    st.session_state.enable_speech = enable_speech
    
    # Add voice selection
    if enable_speech:
        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        selected_voice = st.selectbox("Select Voice", voice_options, 
                                     index=voice_options.index(st.session_state.selected_voice) if st.session_state.selected_voice in voice_options else 4)
        st.session_state.selected_voice = selected_voice
    
    # Clear button in sidebar
    st.markdown("---")
    if st.button("🧹 Clear Everything", help="Clear all data and start fresh"):
        clear_all()
        st.rerun()  # This is the only place we need to rerun

# Initialize prev_url in session state if not present
if 'prev_url' not in st.session_state:
    st.session_state.prev_url = ""
if 'should_transcribe' not in st.session_state:
    st.session_state.should_transcribe = False

# Top section - Video input and transcription
st.header("Video Processing")

# Video source selection - YouTube or Local File
source_col1, source_col2 = st.columns([3, 1])

with source_col1:
    # YouTube URL input
    youtube_url = st.text_input(
        "YouTube URL or leave empty to use local file", 
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url",
        on_change=on_url_change
    )

with source_col2:
    # Local file upload button
    uploaded_file = st.file_uploader(
        "Or upload a video", 
        type=["mp4", "mov", "avi", "mkv", "mpeg4"], 
        key="uploaded_video",
        on_change=on_file_change
    )

# Show a divider
st.markdown("---" if youtube_url or uploaded_file else "")

# Only process the video info if we have a URL and it hasn't been processed
youtube_url = st.session_state.youtube_url
uploaded_file = st.session_state.get("uploaded_video")
transcription_complete = False

if youtube_url and is_youtube_url(youtube_url):
    try:
        # Get video info only if needed
        if not st.session_state.video_info:
            info = get_video_info(youtube_url)
            st.session_state.video_info = info
            video_title = info.get('title', 'Unknown Title')
            st.session_state.video_title = video_title
        else:
            info = st.session_state.video_info
        
        # Display video info
        display_video_info(info)
        
        # Auto-setup for existing transcripts (only if not already done)
        if transcript_exists(youtube_url, OUTPUT_DIR) and not st.session_state.auto_setup_done:
            # Automatically set up RAG for existing transcript
            transcript = read_transcript(youtube_url, OUTPUT_DIR)
            st.session_state.transcript = transcript
            
            # Check API key
            api_key = get_api_key()
            if not api_key:
                st.error("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")
            else:
                try:
                    with st.spinner("Automatically initializing RAG system..."):
                        chunk_count = init_rag_system(transcript, st.session_state.video_title)
                        st.success(f"Video processed and is ready for conversation and generating other contents.")
                        st.session_state.auto_setup_done = True
                        transcription_complete = True
                except ValueError as e:
                    st.error(str(e))
    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
elif uploaded_file:
    # Display local file info
    st.subheader("Local Video Preview")
    st.info(f"File: {uploaded_file.name}")
    
    # If we have a size, display it
    if hasattr(uploaded_file, 'size'):
        size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"Size: {size_mb:.2f} MB")
    
    # Set the video title if not already set
    if not st.session_state.video_title:
        st.session_state.video_title = uploaded_file.name

# Determine if we have valid input
has_yt_url = youtube_url and is_youtube_url(youtube_url)
has_local_file = uploaded_file is not None

# Only enable the button if we have a valid input
transcribe_button = st.button(
    "Process Video", 
    disabled=not (has_yt_url or has_local_file),
    on_click=on_transcribe_click
)

# Only run transcription if the button was just clicked
if st.session_state.should_transcribe:
    st.session_state.should_transcribe = False  # Reset the flag
    
    try:
        # Reset auto setup flag if we're manually transcribing
        st.session_state.auto_setup_done = False
        
        # Check API key
        api_key = get_api_key()
        if not api_key:
            st.error("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")
        else:
            st.session_state.processing = True
            
            # Process different types of input
            if has_yt_url:
                # Check if transcript already exists
                transcript = None
                transcript_path = None
                
                if transcript_exists(youtube_url, OUTPUT_DIR):
                    # If transcript exists, read it
                    transcript = read_transcript(youtube_url, OUTPUT_DIR)
                    st.session_state.transcript = transcript
                    
                    # Get the transcript path for download link
                    video_id = youtube_url.split("v=")[1].split("&")[0] if "v=" in youtube_url else youtube_url.split("/")[-1]
                    transcript_path = os.path.join(OUTPUT_DIR, f"{video_id}_transcript.txt")
                    
                    st.success("Using existing transcript!")
                else:
                    # If no transcript exists, process the video
                    # Display progress
                    download_progress = st.progress(0)
                    download_status = st.empty()
                    download_status.text("Downloading audio from YouTube...")
                    
                    # Transcription progress bar (create but keep empty until needed)
                    transcribe_progress = st.progress(0)
                    transcribe_status = st.empty()
                    
                    # Create output directory
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    
                    # Download the audio
                    download_status.text("Downloading audio...")
                    audio_path = process_video_download(youtube_url, OUTPUT_DIR)
                    download_progress.progress(100)
                    download_status.text("Download complete!")
                    
                    # Transcribe the audio
                    transcribe_status.text("Transcribing audio...")
                    transcribe_progress.progress(10)
                    
                    # Process transcription
                    transcript, transcript_path = process_video_transcribe(
                        audio_path, OUTPUT_DIR, api_key,
                        progress_callback=lambda p: transcribe_progress.progress(p),
                        model=st.session_state.transcription_model
                    )
                    st.session_state.transcript = transcript
                    # Store results
                    result = {
                        'transcript': transcript,
                        'transcript_path': transcript_path,
                        'audio_path': audio_path
                    }
                    
                    # Update final progress
                    transcribe_progress.progress(100)
                    transcribe_status.text("Transcription complete!")
                    
                    # Display results
                    time.sleep(1)
                    
                    # Remove progress indicators
                    download_status.empty()
                    download_progress.empty()
                    transcribe_status.empty()
                    transcribe_progress.empty()
                    
                    # Show results
                    st.success("Transcription complete!")
                    
                    # Display download links
                    st.subheader("Download Files")
                    transcript_link = get_download_link(result['transcript_path'], "Download Transcript (TXT)")
                    st.markdown(transcript_link, unsafe_allow_html=True)
                    os.remove(result['audio_path'])
                    
                # Set the video title if not already set
                if not st.session_state.video_title:
                    # Try to get title from YouTube
                    try:
                        info = get_video_info(youtube_url)
                        st.session_state.video_title = info.get('title', 'YouTube Video')
                    except:
                        st.session_state.video_title = "YouTube Video"
                
            elif has_local_file:
                # Local file processing
                # Display progress for transcription (no download needed)
                transcribe_progress = st.progress(0)
                transcribe_status = st.empty()
                transcribe_status.text("Processing local video file...")
                
                # Create output directory
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                # Save the uploaded file to disk
                file_extension = uploaded_file.name.split('.')[-1].lower()
                temp_file_path = os.path.join(OUTPUT_DIR, f"local_upload_{int(time.time())}.{file_extension}")
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                transcribe_status.text("Extracting audio...")
                transcribe_progress.progress(10)
                
                # Extract audio using FFmpeg if it's a video file
                video_id = f"local_{int(time.time())}"
                audio_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp3")
                
                ffmpeg_path, _ = verify_ffmpeg()
                subprocess.run([
                    ffmpeg_path, '-i', temp_file_path, '-q:a', '0', '-map', 'a', 
                    audio_path, '-y'
                ], check=True, capture_output=True)

                os.remove(temp_file_path)
                transcribe_status.text("Transcribing audio...")
                transcribe_progress.progress(30)
                
                # Transcribe the audio
                transcript, transcript_path = process_video_transcribe(
                    audio_path, OUTPUT_DIR, api_key,
                    progress_callback=lambda p: transcribe_progress.progress(min(30 + int(p * 0.7), 100)),
                    model=st.session_state.transcription_model 
                )
                st.session_state.transcript = transcript
                
                # Store results
                result = {
                    'transcript': transcript,
                    'transcript_path': transcript_path,
                    'audio_path': audio_path
                }
                
                # Update final progress
                transcribe_progress.progress(100)
                transcribe_status.text("Transcription complete!")
                
                # Display results
                time.sleep(1)
                
                # Remove progress indicators
                transcribe_status.empty()
                transcribe_progress.empty()
                
                # Show results
                st.success("Transcription complete!")
                
                # Display download links
                st.subheader("Download Files")
                transcript_link = get_download_link(result['transcript_path'], "Download Transcript (TXT)")
                st.markdown(transcript_link, unsafe_allow_html=True)

                # Delete audio file 
                os.remove(result['audio_path'])
                
                # Set the video title
                st.session_state.video_title = uploaded_file.name
            
            # Now that we have a transcript (either from YouTube or local file),
            # initialize RAG system
            try:
                with st.spinner("Initializing RAG system..."):
                    video_title = st.session_state.video_title
                    chunk_count = init_rag_system(transcript, video_title)
                    st.success(f"Video processed and is ready for conversation and generating additional contents.")
                    st.session_state.auto_setup_done = True
                    transcription_complete = True
            except ValueError as e:
                st.error(str(e))
            
            # Update processing state
            st.session_state.processing = False
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)  # This shows the full traceback
        st.session_state.processing = False

# Horizontal line to separate sections
st.markdown("---")

# Middle section - Interactive chat and quiz side by side
if st.session_state.rag_initialized:
    # Add buttons for different outputs above the columns
    st.subheader("Generate Additional Content")
    
    output_buttons = st.columns([1, 1, 1, 1])
    with output_buttons[0]:
        if st.button("📋 Hierarchical Summary", use_container_width=True):
            show_summary_popup()
    with output_buttons[1]:
        if st.button("🎲 Generate Quiz", use_container_width=True):
            generate_quiz()
            st.rerun()
    
    with output_buttons[2]:
        if st.button("🔤 Flashcards", use_container_width=True):
            # Hide quiz when showing flashcards
            st.session_state.show_quiz = False
            generate_flashcards()
            st.rerun()

    # We can add more content generation buttons here in columns 2 and 3
    
    # Display the summary popup if it should be shown
    if st.session_state.show_summary_popup and st.session_state.hierarchical_summary:
        popup_html = st.session_state.summary_generator.create_summary_popup_html(
            st.session_state.hierarchical_summary
        )
        components.html(popup_html, height=600, scrolling=True)

        # Add a button to close the popup
        if st.button("Close Summary", key="close_summary_button"):
            st.session_state.show_summary_popup = False
            st.rerun()
    
    # Create a two-column layout for chat and quiz
    chat_col, quiz_col = st.columns([1, 1])
    
    # Left column - Chat interface
    with chat_col:
        st.subheader(f"💬 Chat with Video: {st.session_state.video_title}")
        
        # Create a container with fixed height and scrolling for messages only if there are messages
        if len(st.session_state.chat_history) > 0:
            
            # Chat messages container
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"<div style='background-color: #E1F5FE; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #F1F8E9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><b>Video Assistant:</b> {message['content']}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add speech button if speech is enabled
        if st.session_state.enable_speech:
            # Add a speak button
            if st.button("🎤 Speak (Max. 5 seconds)", key="speak_button"):
                handle_speech_input()
        
        # Question input with direct callback
        st.text_input(
            "Ask a question about the video:", 
            key="question_input",
            on_change=handle_question_submission,
            placeholder="Type your question here and press Enter"
        )
    
    # Right column - Quiz interface or Flashcards
    with quiz_col:
        # Show flashcards if they're active
        if st.session_state.show_flashcards:
            st.header("Flashcards")
            
            if st.session_state.flashcards and len(st.session_state.flashcards) > 0:
                # Create a container for the flashcard
                flashcard_container = st.container()
                
                # Get the current flashcard
                current_index = st.session_state.current_flashcard_index
                # Ensure the index is valid
                if current_index >= len(st.session_state.flashcards):
                    current_index = 0
                    st.session_state.current_flashcard_index = 0
                    
                current_card = st.session_state.flashcards[current_index]
                total_cards = len(st.session_state.flashcards)
                
                # Display card count
                st.markdown(f"**Card {current_index + 1} of {total_cards}**")
                
                # Display flashcard in a styled container
                with flashcard_container:
                    # Card styling
                    st.markdown("""
                    <style>
                    .flashcard {
                        background-color: #f9f9f9;
                        border-radius: 10px;
                        border: 1px solid #ddd;
                        padding: 20px;
                        min-height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        text-align: center;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        font-size: 18px;
                    }
                    .flashcard-front {
                        font-weight: bold;
                        font-size: 20px;
                        padding: 20px;
                    }
                    .flashcard-back {
                        margin-top: 15px;
                        padding: 20px;
                        border-top: 1px dashed #ccc;
                        background-color: #efefef;
                        border-radius: 0 0 10px 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Verify card has front and back properties
                    if "front" in current_card and "back" in current_card:
                        # Display front of card
                        st.markdown(f'<div class="flashcard"><div class="flashcard-front">{current_card["front"]}</div>', unsafe_allow_html=True)
                        
                        # Display back of card if toggled
                        if st.session_state.show_flashcard_answer:
                            st.markdown(f'<div class="flashcard-back">{current_card["back"]}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Invalid flashcard format. Card should have 'front' and 'back' properties.")
                
                # Navigation buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("⬅️ Previous", on_click=prev_flashcard, key="prev_flashcard_btn", use_container_width=True)
                with col2:
                    flip_label = "Hide Answer" if st.session_state.show_flashcard_answer else "Show Answer"
                    st.button(flip_label, on_click=toggle_answer, key="toggle_flashcard_btn", use_container_width=True)
                with col3:
                    st.button("Next ➡️", on_click=next_flashcard, key="next_flashcard_btn", use_container_width=True)
                
                # Add spacer
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Additional controls in a row
                control_cols = st.columns(2)
                with control_cols[0]:
                    if st.button("Generate New Flashcards", key="new_flashcards_button", use_container_width=True):
                        generate_flashcards()
                        st.rerun()
                
                with control_cols[1]:
                    if st.button("Switch to Quiz", key="switch_to_quiz_button", use_container_width=True):
                        st.session_state.show_flashcards = False
                        if 'quiz' in st.session_state and st.session_state.quiz:
                            st.session_state.show_quiz = True
                        else:
                            generate_quiz()
                        st.rerun()
                
            else:
                st.error("No flashcards available. Please try generating them again.")
                
                if st.button("Generate Flashcards Now", key="try_flashcards_again"):
                    generate_flashcards()
                    st.rerun()
                    
        # Show quiz if it's active
        elif st.session_state.show_quiz:
            st.header("Quiz")
            
            if 'quiz' in st.session_state and st.session_state.quiz:
                # Current question details
                current_q_index = st.session_state.current_quiz_question
                
                if not st.session_state.quiz_completed and current_q_index < len(st.session_state.quiz):
                    current_question = st.session_state.quiz[current_q_index]
                    
                    # Progress indicator
                    st.progress((current_q_index) / len(st.session_state.quiz))
                    st.write(f"Question {current_q_index + 1} of {len(st.session_state.quiz)}")
                    
                    # Display the question
                    st.subheader(current_question["question"])
                    
                    # Check if we've already answered this question
                    question_answered = False
                    current_result = None
                    
                    for result in st.session_state.quiz_results:
                        if result.get("question_index") == current_q_index:
                            question_answered = True
                            current_result = result
                            break
                    
                    if not question_answered:
                        # Display options in a form
                        with st.form(key=f"quiz_form_{current_q_index}"):
                            # Get options
                            options = current_question.get("options", [])
                            
                            if options:
                                # Create radio options with just the text, but use the letter as the value
                                option_dict = {letter: text for letter, text in options}
                                
                                # Store user selection (the letter only)
                                selected_letter = st.radio(
                                    "Select your answer:",
                                    options=[letter for letter, _ in options],
                                    format_func=lambda x: f"{x}: {option_dict[x]}",
                                    key=f"quiz_q{current_q_index}"
                                )
                                
                                # Submit button
                                submitted = st.form_submit_button("Submit Answer")
                                
                                if submitted:
                                    # Check if answer is correct - handle missing 'answer' key
                                    correct_letter = current_question.get("correct", "")
                                    is_correct = selected_letter == correct_letter
                                    
                                    # Store the result
                                    result = {
                                        "question_index": current_q_index,
                                        "question": current_question["question"],
                                        "selected": selected_letter,
                                        "correct": is_correct,
                                        "answer": correct_letter,
                                        "explanation": current_question.get("explanation", "")
                                    }
                                    
                                    st.session_state.quiz_results.append(result)
                                    st.rerun()
                    else:
                        # Show the answer result for the current question
                        # This displays after submission and before moving to next question
                        st.subheader("Your Answer:")
                        
                        if current_result["correct"]:
                            st.success(f"✅ Correct! You selected: {current_result['selected']}")
                        else:
                            st.error(f"❌ Incorrect. You selected: {current_result['selected']}")
                            st.info(f"The correct answer is: {current_result['answer']}")
                        
                        # Show explanation if available
                        if current_result.get("explanation"):
                            st.markdown("**Explanation:**")
                            st.markdown(current_result["explanation"])
                        
                        # Next button
                        next_col1, next_col2 = st.columns([3, 1])
                        with next_col2:
                            if st.button("Next Question", key=f"next_q_{current_q_index}", use_container_width=True):
                                # Move to next question or complete quiz
                                if current_q_index + 1 < len(st.session_state.quiz):
                                    st.session_state.current_quiz_question += 1
                                else:
                                    st.session_state.quiz_completed = True
                                st.rerun()
                
                # Show results if the quiz is completed
                elif st.session_state.quiz_completed:
                    st.subheader("Quiz Results")
                    
                    # Calculate score
                    correct_answers = sum(1 for result in st.session_state.quiz_results if result.get("correct", False))
                    total_questions = len(st.session_state.quiz_results)
                    score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
                    
                    # Create an engaging score display with a progress bar
                    score_col1, score_col2 = st.columns([1, 1])
                    with score_col1:
                        st.markdown(f"### Your Score: {correct_answers}/{total_questions}")
                        
                        # Visual score with progress bar
                        st.progress(score_percentage / 100)
                        
                    with score_col2:
                        # Display a message based on score
                        if score_percentage >= 80:
                            st.markdown("### 🎉 Excellent!")
                            st.success("You have a strong understanding of the content!")
                        elif score_percentage >= 60:
                            st.markdown("### 👍 Good Job!")
                            st.info("You've grasped many of the key concepts.")
                        else:
                            st.markdown("### 📚 Keep Learning")
                            st.warning("You might want to review the video again.")
                    
                    # Interactive summary - visual representation of correct vs incorrect
                    st.subheader("Performance Summary")
                    
                    # Calculate statistics
                    correct_count = correct_answers
                    incorrect_count = total_questions - correct_answers
                    
                    # Create interactive summary with HTML/CSS
                    summary_html = f"""
                    <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
                        <div style="display: flex; margin-bottom: 15px;">
                            <div style="flex: {max(1, correct_count)}; background-color: #4CAF50; height: 30px; border-radius: 5px 0 0 5px; color: white; display: flex; align-items: center; justify-content: center;">
                                {correct_count} Correct
                            </div>
                            <div style="flex: {max(1, incorrect_count)}; background-color: #F44336; height: 30px; border-radius: 0 5px 5px 0; color: white; display: flex; align-items: center; justify-content: center;">
                                {incorrect_count} Incorrect
                            </div>
                        </div>
                        <div style="text-align: center; font-size: 18px; font-weight: bold;">
                            {score_percentage:.1f}% Correct
                        </div>
                    </div>
                    """
                    
                    st.markdown(summary_html, unsafe_allow_html=True)
                    
                    # Display detailed results for each question
                    st.subheader("Question Review")
                    
                    for i, result in enumerate(st.session_state.quiz_results):
                        # Create expandable section for each question
                        with st.expander(f"Question {i+1}: {result['question']}"):
                            # Show if answer was correct/incorrect
                            if result.get("correct", False):
                                st.markdown("✅ **Correct!**")
                            else:
                                st.markdown("❌ **Incorrect**")
                            
                            # Show selected answer
                            st.markdown(f"You selected: **{result.get('selected', 'No answer')}**")
                            
                            # Show correct answer if wrong
                            if not result.get("correct", False):
                                st.markdown(f"Correct answer: **{result.get('answer', 'Unknown')}**")
                            
                            # Show explanation if available
                            if result.get("explanation"):
                                st.markdown("**Explanation:**")
                                st.markdown(result["explanation"])
                    
                    # Buttons for next actions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Take New Quiz", use_container_width=True):
                            # Reset quiz state and generate a new quiz
                            st.session_state.quiz_results = []
                            st.session_state.current_quiz_question = 0
                            st.session_state.quiz_completed = False
                            generate_quiz()
                            st.rerun()
                    
                    with col2:
                        if st.button("Switch to Flashcards", key="switch_to_flashcards", use_container_width=True):
                            st.session_state.show_quiz = False
                            if st.session_state.flashcards:
                                st.session_state.show_flashcards = True
                            else:
                                generate_flashcards()
                            st.rerun()
            
            else:
                st.error("No quiz available. Please try generating a quiz.")
                
                if st.button("Generate Quiz Now", key="gen_quiz_now"):
                    generate_quiz()
                    st.rerun()
        
        # Default state - no quiz or flashcards showing
        else:
            st.subheader("🧠 Interactive Learning")

else:
    # Show a message when RAG is not initialized
    st.info("Transcribe a video to start chatting and create additional content!")

# Mark app as initialized
st.session_state.initialized = True
