import os
import yt_dlp
from openai import OpenAI
from pathlib import Path
import subprocess
import sys
import shutil
import os
import math
from pydub import AudioSegment
from openai import OpenAI

# Hard-coded FFmpeg location - update this to your specific path
FFMPEG_LOCATION = r'ffmpeg-2025-03-10-git-87e5da9067-essentials_build\ffmpeg-2025-03-10-git-87e5da9067-essentials_build\bin'
 
_video_info_cache = {}
_video_id_cache = {}

def process_video_download(youtube_url, output_dir):
    """
    Download audio from a YouTube video
    
    Args:
        youtube_url: URL of the YouTube video
        output_dir: Directory to save the output
        
    Returns:
        str: Path to the downloaded audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract video ID from URL
    video_id = None
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Could not extract video ID from URL")
    
    # Set output paths
    audio_path = os.path.join(output_dir, f"{video_id}.mp3")
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, f"{video_id}"),
        'quiet': True
    }
    
    # Download audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # Verify audio file exists
    if not os.path.exists(audio_path):
        # Try with an extension that yt-dlp might have used
        potential_paths = [
            os.path.join(output_dir, f"{video_id}.mp3"),
            os.path.join(output_dir, f"{video_id}.m4a"),
            os.path.join(output_dir, f"{video_id}.webm")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                # Convert to mp3 if it's not already
                if not path.endswith('.mp3'):
                    ffmpeg_path = verify_ffmpeg()[0]
                    output_mp3 = os.path.join(output_dir, f"{video_id}.mp3")
                    subprocess.run([
                        ffmpeg_path, '-i', path, '-c:a', 'libmp3lame', 
                        '-q:a', '2', output_mp3, '-y'
                    ], check=True, capture_output=True)
                    os.remove(path)  # Remove the original file
                    audio_path = output_mp3
                else:
                    audio_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find downloaded audio file for video {video_id}")
    
    return audio_path


def process_video_transcribe(audio_path, output_dir, api_key, progress_callback=None, model="gpt-4o-transcribe"):
    """
    Transcribe an audio file using OpenAI API, with automatic chunking for large files
    Always uses the selected model, with no fallback
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the transcript
        api_key: OpenAI API key
        progress_callback: Function to call with progress updates (0-100)
        model: The model to use for transcription (default: gpt-4o-transcribe)
        
    Returns:
        tuple: (transcript text, transcript path)
    """
    # Extract video ID from audio path
    video_id = os.path.basename(audio_path).split('.')[0]
    transcript_path = os.path.join(output_dir, f"{video_id}_transcript.txt")
    
    # Setup OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Update progress
    if progress_callback:
        progress_callback(10)
    
    # Get file size in MB
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    
    # Universal chunking thresholds - apply to both models
    max_size_mb = 25  # 25MB chunk size for both models
    max_duration_seconds = 1500  # 1500 seconds chunk duration for both models
    
    # Load the audio file to get its duration
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000  # pydub uses milliseconds
    except Exception as e:
        print(f"Error loading audio to check duration: {e}")
        audio = None
        duration_seconds = 0
    
    # Determine if chunking is needed
    needs_chunking = False
    chunking_reason = []
    
    if file_size_mb > max_size_mb:
        needs_chunking = True
        chunking_reason.append(f"size ({file_size_mb:.2f}MB exceeds {max_size_mb}MB)")
    
    if duration_seconds > max_duration_seconds:
        needs_chunking = True
        chunking_reason.append(f"duration ({duration_seconds:.2f}s exceeds {max_duration_seconds}s)")
    
    # Log the decision
    if needs_chunking:
        reason_str = " and ".join(chunking_reason)
        print(f"Audio needs chunking due to {reason_str}. Using {model} for transcription.")
    else:
        print(f"Audio file is within limits. Using {model} for direct transcription.")
    
    # Check if file needs chunking
    if needs_chunking:
        if progress_callback:
            progress_callback(15)
        
        # Split the audio file into chunks and transcribe each chunk using the selected model only
        full_transcript = split_and_transcribe(
            audio_path, client, model, progress_callback, 
            max_size_mb, max_duration_seconds, audio
        )
    else:
        # File is small enough, transcribe directly with the selected model
        with open(audio_path, "rb") as audio_file:
            if progress_callback:
                progress_callback(30)
                
            transcript_response = client.audio.transcriptions.create(
                model=model, 
                file=audio_file
            )
            
            if progress_callback:
                progress_callback(80)
            
            full_transcript = transcript_response.text
    
    # Save transcript to file
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    
    # Update progress
    if progress_callback:
        progress_callback(100)
    
    return full_transcript, transcript_path

def split_and_transcribe(audio_path, client, model, progress_callback=None, 
                         max_size_mb=25, max_duration_seconds=1500, audio=None):
    """
    Split an audio file into chunks and transcribe each chunk using only the selected model
    
    Args:
        audio_path: Path to the audio file
        client: OpenAI client
        model: Model to use for transcription (will not fall back to other models)
        progress_callback: Function to call with progress updates
        max_size_mb: Maximum file size in MB
        max_duration_seconds: Maximum duration in seconds
        audio: Pre-loaded AudioSegment (optional)
        
    Returns:
        str: Combined transcript from all chunks
    """
    # Load the audio file if not provided
    if audio is None:
        audio = AudioSegment.from_file(audio_path)
    
    # Get audio duration in seconds
    duration_seconds = len(audio) / 1000
    
    # Calculate the number of chunks needed based on both size and duration
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    
    chunks_by_size = math.ceil(file_size_mb / (max_size_mb * 0.9))  # Use 90% of max to be safe
    chunks_by_duration = math.ceil(duration_seconds / (max_duration_seconds * 0.95))  # Use 95% of max to be safe
    
    num_chunks = max(chunks_by_size, chunks_by_duration)
    
    print(f"Splitting audio into {num_chunks} chunks based on size ({chunks_by_size}) and duration ({chunks_by_duration})")
    
    # Calculate chunk duration in milliseconds
    chunk_length_ms = len(audio) // num_chunks
    
    # Create temp directory for chunks if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(audio_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split the audio into chunks and transcribe each chunk
    transcripts = []
    
    for i in range(num_chunks):
        if progress_callback:
            # Update progress: 20% for splitting, 60% for transcribing
            progress_percent = 20 + int((i / num_chunks) * 60)
            progress_callback(progress_percent)
        
        # Calculate start and end times for this chunk
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, len(audio))
        
        # Extract the chunk
        chunk = audio[start_ms:end_ms]
        
        # Save the chunk to a temporary file
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        
        # Log chunk information
        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        chunk_duration = len(chunk) / 1000
        print(f"Chunk {i+1}/{num_chunks}: {chunk_size_mb:.2f}MB, {chunk_duration:.2f}s")
        
        # Transcribe the chunk using ONLY the selected model (no fallback)
        try:
            with open(chunk_path, "rb") as chunk_file:
                transcript_response = client.audio.transcriptions.create(
                    model=model,
                    file=chunk_file
                )
                
                # Add to our list of transcripts
                transcripts.append(transcript_response.text)
        except Exception as e:
            print(f"Error transcribing chunk {i+1} with {model}: {e}")
            # Add a placeholder for the failed chunk
            transcripts.append(f"[Transcription failed for segment {i+1}]")
        
        # Clean up the temporary chunk file
        os.remove(chunk_path)
    
    # Clean up the temporary directory
    try:
        os.rmdir(temp_dir)
    except:
        print(f"Note: Could not remove temporary directory {temp_dir}")
    
    # Combine all transcripts with proper spacing
    full_transcript = " ".join(transcripts)
    
    return full_transcript

def get_video_info(youtube_url):
    """Get video information without downloading."""
    # Check local cache first
    global _video_info_cache
    if youtube_url in _video_info_cache:
        return _video_info_cache[youtube_url]
        
    # Extract info if not cached
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        # Cache the result
        _video_info_cache[youtube_url] = info
        # Also cache the video ID separately
        _video_id_cache[youtube_url] = info.get('id', 'video')
        return info

def get_video_id(youtube_url):
    """Get just the video ID without re-extracting if already known."""
    global _video_id_cache
    if youtube_url in _video_id_cache:
        return _video_id_cache[youtube_url]
    
    # If not in cache, extract from URL directly if possible
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        _video_id_cache[youtube_url] = video_id
        return video_id
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        _video_id_cache[youtube_url] = video_id
        return video_id
    
    # If we can't extract directly, fall back to full info extraction
    info = get_video_info(youtube_url)
    video_id = info.get('id', 'video')
    return video_id

def get_transcript_path(youtube_url, output_dir):
    """Get the expected transcript path for a given YouTube URL."""
    # Get video ID with caching
    video_id = get_video_id(youtube_url)
    # Return expected transcript path
    return os.path.join(output_dir, f"{video_id}_transcript.txt")

def transcript_exists(youtube_url, output_dir):
    """Check if a transcript already exists for this video."""
    transcript_path = get_transcript_path(youtube_url, output_dir)
    return os.path.exists(transcript_path)

def verify_ffmpeg():
    """Verify that FFmpeg is available and print its location."""
    # Add FFmpeg to PATH
    os.environ['PATH'] = FFMPEG_LOCATION + os.pathsep + os.environ['PATH']
    
    # Check if FFmpeg binaries exist
    ffmpeg_path = os.path.join(FFMPEG_LOCATION, 'ffmpeg.exe')
    ffprobe_path = os.path.join(FFMPEG_LOCATION, 'ffprobe.exe')
    
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"FFmpeg executable not found at: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path):
        raise FileNotFoundError(f"FFprobe executable not found at: {ffprobe_path}")
    
    print(f"FFmpeg found at: {ffmpeg_path}")
    print(f"FFprobe found at: {ffprobe_path}")
    
    # Try to execute FFmpeg to make sure it works
    try:
        # Add shell=True for Windows and capture errors properly
        result = subprocess.run([ffmpeg_path, '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               shell=True,  # This can help with permission issues on Windows
                               check=False)
        
        if result.returncode == 0:
            print(f"FFmpeg version: {result.stdout.decode().splitlines()[0]}")
        else:
            error_msg = result.stderr.decode()
            print(f"FFmpeg error: {error_msg}")
            
            # Check for specific permission errors
            if "Access is denied" in error_msg:
                print("Permission error detected. Trying alternative approach...")
                
                # Try an alternative approach - just check file existence without execution
                if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
                    print("FFmpeg files exist but execution test failed due to permissions.")
                    print("WARNING: The app may fail when trying to process videos.")
                    # Return paths anyway and hope for the best when actually used
                    return ffmpeg_path, ffprobe_path
                
            raise RuntimeError(f"FFmpeg execution failed: {error_msg}")
    except Exception as e:
        print(f"Error checking FFmpeg: {e}")
        
        # Fallback option if verification fails but files exist
        if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
            print("WARNING: FFmpeg files exist but verification failed.")
            print("Attempting to continue anyway, but video processing may fail.")
            return ffmpeg_path, ffprobe_path
            
        raise
    
    return ffmpeg_path, ffprobe_path

def download_audio(youtube_url, output_path='audio.mp3'):
    """Download audio from a YouTube video."""
    print(f"Downloading audio from {youtube_url}...")
    
    # Verify and get FFmpeg paths - but don't stop on verification issues
    try:
        ffmpeg_path, ffprobe_path = verify_ffmpeg()
    except Exception as e:
        print(f"WARNING: FFmpeg verification failed: {e}")
        print("Attempting to continue with download anyway...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.mp3', ''),  # yt-dlp adds extension automatically
        'ffmpeg_location': FFMPEG_LOCATION,
        'ignoreerrors': True,  # Try to continue on errors
    }
    
    # Print some debug info
    print(f"Using FFmpeg location: {FFMPEG_LOCATION}")
    print(f"Output path: {output_path}")
    print(f"PATH environment variable: {os.environ['PATH']}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        print(f"Error during download: {e}")
        raise
    
    print(f"Audio downloaded to {output_path}")
    return output_path

def transcribe_audio(audio_path, api_key, model="gpt-4o-transcribe"):
    """Transcribe audio using OpenAI's Whisper or gpt-4o-transcribe API."""
    print(f"Transcribing audio with OpenAI's {model} model...")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    
    return response.text

def save_transcript(transcript, output_path='transcript.txt'):
    """Save transcript to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f"Transcript saved to {output_path}")
    return output_path


def process_video(youtube_url, output_dir, api_key, model="gpt-4o-transcribe"):
    """
    Process a YouTube video to generate a transcript
    Wrapper function that combines download and transcription
    
    Args:
        youtube_url: URL of the YouTube video
        output_dir: Directory to save the output
        api_key: OpenAI API key
        model: The model to use for transcription (default: gpt-4o-transcribe)
        
    Returns:
        dict: Dictionary containing transcript and file paths
    """
    # First download the audio
    print("Downloading video...")
    audio_path = process_video_download(youtube_url, output_dir)
    
    print("Transcribing video...")
    # Then transcribe the audio
    transcript, transcript_path = process_video_transcribe(audio_path, output_dir, api_key, model=model)
    
    # Return the combined results
    return {
        'transcript': transcript,
        'transcript_path': transcript_path,
        'audio_path': audio_path
    }

def read_transcript(youtube_url, output_dir):
    """Read existing transcript for a video."""
    transcript_path = get_transcript_path(youtube_url, output_dir)
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None
