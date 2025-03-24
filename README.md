# Talk-to-Videos

An AI-powered video exploration and learning tool that allows you to interact with educational videos through conversation, quizzes, and flashcards.

[Workflow of Talk-to-Videos](images/workflow.png)

## Introduction

Talk-to-Videos transforms the way you learn from video content by creating an interactive experience powered by AI. The application transcribes videos, builds a specialized Retrieval Augmented Generation (RAG) system, and enables you to:

- Ask questions about the video content and receive accurate answers
- Generate comprehensive hierarchical summaries
- Create interactive quizzes to test your knowledge
- Study with auto-generated flashcards
- Use voice input and output for a hands-free experience

Perfect for students, educators, researchers, and anyone who wants to extract maximum value from educational video content.

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (required for video processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/umairalipathan1980/Talk-to-Videos.git
cd Talk-to-Videos
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download and Install FFmpeg

1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the downloaded archive to the main folder of the Talk-to-Videos repository

### Step 4: Set up OpenAI API Key

Create a `.streamlit/secrets.toml` file with your OpenAI API key:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

## Features

### Video Processing and Transcription

Upload local video files or provide YouTube URLs to automatically transcribe and analyze content. The application supports:

- YouTube video downloads
- Local video file uploads
- Multiple transcription models
- Transcript downloads

[Main UI with Video Preview](images/fig1.png)

### Intelligent Conversation Interface

After transcription, the application creates a RAG system that enables you to ask questions about any aspect of the video content.

[Conversation Ready Interface](images/fig1_1.png)

Key capabilities:
- Natural language Q&A about video content
- Voice-enabled conversation (optional)
- Multiple voice options for responses
- Contextual understanding of video content

[Example Chat with Video](images/fig2.png)

### Learning Tools

#### Hierarchical Summary

Generate a structured, multi-level summary that organizes the video content into main topics and subtopics, helping you understand the overall structure.

#### Interactive Quizzes

Test your understanding with automatically generated quizzes based on the video content.

- Multiple-choice questions covering key concepts
- Detailed explanations for each answer
- Performance tracking and scoring
- Visual results analysis

[Quiz Results](images/fig3.png)

#### Flashcards

Study effectively with AI-generated flashcards derived from the video content.

- Front (question) and back (answer) format
- Navigation controls (previous, next)
- Answer reveal functionality
- Generate new sets as needed

[Flashcards](images/fig4.png)

## Speech Capabilities

Enable speech input and output for a hands-free experience:

- Record questions with your microphone
- Hear answers spoken back to you
- Choose from multiple voice options
- Seamless text and speech integration

## Technical Details

The application combines several advanced technologies:

- **Retrieval Augmented Generation (RAG)**: Creates a vector database from video transcripts
- **OpenAI API Integration**: Powers the conversation, quiz generation, and flashcards
- **FFmpeg**: Handles video and audio processing
- **Streamlit**: Provides the interactive web interface
- **Speech-to-Text and Text-to-Speech**: Enables voice interaction

## Usage Tips

1. **Transcription Models**: Choose between different models based on your needs:
   - `whisper-1`: Faster, supports up to 25MB files
   - `gpt-4o-transcribe`: Higher accuracy, supports up to 50MB files

2. **Keeping Audio**: Toggle the option to keep or delete the extracted audio after transcription

3. **Speech Settings**: Enable speech input/output and select your preferred voice in the sidebar

4. **Clearing Data**: Use the "Clear Everything" button in the sidebar to reset the application state

## License

MIT License

## Acknowledgments

- This project uses OpenAI's APIs for natural language processing and speech capabilities
- Built with Streamlit for a responsive and interactive user interface
- Video processing powered by FFmpeg
