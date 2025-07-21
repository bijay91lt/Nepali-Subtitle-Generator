import streamlit as st
import torch
import librosa
import tempfile
import os
import io
import subprocess
from datetime import timedelta
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Nepali Audio/Video Transcription",
    page_icon="🎬",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_resource
def load_model_and_processor():
    """
    Load the Whisper model and processor with caching for better performance.
    """
    MODEL_NAME = "openai/whisper-small"
    FINE_TUNED_MODEL_PATH = "final_model(5308)"
    DEVICE = torch.device("cpu")
    
    with st.spinner("Loading model and processor..."):
        # Load Processor
        try:
            processor = WhisperProcessor.from_pretrained(FINE_TUNED_MODEL_PATH)
            st.success(f"Processor loaded from: {FINE_TUNED_MODEL_PATH}")
        except OSError:
            st.warning(f"Could not load processor from {FINE_TUNED_MODEL_PATH}. Loading from base model: {MODEL_NAME}")
            processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        
        # Load Model
        try:
            model = WhisperForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_PATH)
            model.to(DEVICE)
            model.eval()
            
            # Configure generation settings
            if not hasattr(model, 'generation_config') or model.generation_config is None:
                model.generation_config = GenerationConfig.from_model_config(model.config)
                if model.generation_config is None:
                    model.generation_config = GenerationConfig()
            
            model.generation_config.forced_decoder_ids = None
            model.generation_config.suppress_tokens = []
            
            st.success("Fine-tuned model loaded successfully!")
            return model, processor, DEVICE
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None

def check_ffmpeg():
    """
    Check if ffmpeg is available in the system.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_audio_from_video(video_path):
    """
    Extract audio from video file using ffmpeg.
    """
    try:
        with st.spinner("Extracting audio from video..."):
            # Create temporary audio file
            temp_audio_path = tempfile.mktemp(suffix=".wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,           # Input video file
                '-vn',                      # No video output
                '-acodec', 'pcm_s16le',     # Audio codec
                '-ar', '16000',             # Sample rate
                '-ac', '1',                 # Mono channel
                '-y',                       # Overwrite output file
                temp_audio_path
            ]
            
            # Run ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                st.error(f"FFmpeg error: {result.stderr}")
                return None
            
            # Check if output file was created
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                return temp_audio_path
            else:
                st.error("Failed to extract audio from video")
                return None
                
    except Exception as e:
        st.error(f"Error extracting audio from video: {e}")
        return None

def load_audio_file(file_path, is_video=False):
    """
    Load and preprocess audio file or extract audio from video.
    """
    try:
        if is_video:
            # Extract audio from video first
            audio_path = extract_audio_from_video(file_path)
            if audio_path is None:
                return None, None
            
            # Load the extracted audio
            array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Clean up temporary audio file
            try:
                os.unlink(audio_path)
            except:
                pass
                
            return array, sampling_rate
        else:
            # Direct audio file loading
            array, sampling_rate = librosa.load(file_path, sr=16000)
            return array, sampling_rate
            
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None

def is_video_file(filename):
    """
    Check if the uploaded file is a video file.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp']
    return Path(filename).suffix.lower() in video_extensions

def transcribe_audio_with_timestamps(model, processor, device, audio_array, chunk_length=30):
    """
    Transcribe audio in chunks and return text with approximate timestamps.
    """
    if audio_array is None:
        return None, None
    
    # Calculate chunk size in samples
    sample_rate = 16000
    chunk_samples = chunk_length * sample_rate
    
    transcriptions = []
    timestamps = []
    
    total_chunks = len(range(0, len(audio_array), chunk_samples))
    progress_bar = st.progress(0)
    
    # Process audio in chunks
    for idx, i in enumerate(range(0, len(audio_array), chunk_samples)):
        chunk = audio_array[i:i + chunk_samples]
        
        # Update progress
        progress_bar.progress((idx + 1) / total_chunks)
        
        # Skip very short chunks
        if len(chunk) < sample_rate:  # Less than 1 second
            continue
        
        # Calculate timestamp
        start_time = i / sample_rate
        end_time = min((i + len(chunk)) / sample_rate, len(audio_array) / sample_rate)
        
        # Process chunk
        input_features = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features
        input_features = input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_features,
                generation_config=model.generation_config
            )
        
        # Decode transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if transcription.strip():  # Only add non-empty transcriptions
            transcriptions.append(transcription.strip())
            timestamps.append((start_time, end_time))
    
    progress_bar.progress(1.0)
    return transcriptions, timestamps

def format_timestamp(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d},{milliseconds:03d}"

def create_srt_content(transcriptions, timestamps):
    """
    Create SRT file content from transcriptions and timestamps.
    """
    srt_content = ""
    
    for i, (text, (start_time, end_time)) in enumerate(zip(transcriptions, timestamps), 1):
        start_formatted = format_timestamp(start_time)
        end_formatted = format_timestamp(end_time)
        
        srt_content += f"{i}\n"
        srt_content += f"{start_formatted} --> {end_formatted}\n"
        srt_content += f"{text}\n\n"
    
    return srt_content

def create_plain_text(transcriptions):
    """
    Create plain text file from transcriptions.
    """
    return "\n".join(transcriptions)

def get_video_info(video_path):
    """
    Get basic information about the video file using ffprobe.
    """
    try:
        # Use ffprobe to get video information
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
            
        import json
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream and 'format' in data:
            duration = float(data['format'].get('duration', 0))
            fps = eval(video_stream.get('r_frame_rate', '0/1'))  # Convert fraction to float
            width = video_stream.get('width', 0)
            height = video_stream.get('height', 0)
            
            return {
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}" if width and height else "Unknown"
            }
        
        return None
        
    except Exception as e:
        st.warning(f"Could not extract video info: {e}")
        return None

# --- Streamlit App ---
def main():
    st.title("🎬 Nepali Audio/Video Transcription")
    st.markdown("Upload an audio or video file to get Nepali transcription with downloadable SRT subtitles that work with any media player")
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        st.error("❌ FFmpeg not found!")
        st.markdown("""
        **FFmpeg is required for video processing. Please install it:**
        
        **Windows:**
        1. Download from: https://ffmpeg.org/download.html
        2. Extract and add to system PATH
        3. Or use: `winget install ffmpeg` or `choco install ffmpeg`
        
        **macOS:**
        ```bash
        brew install ffmpeg
        ```
        
        **Ubuntu/Debian:**
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
        
        **After installation, restart the application.**
        """)
        return
    
    # Load model
    model, processor, device = load_model_and_processor()
    
    if model is None:
        st.error("Failed to load the model. Please check your model files.")
        return
    
    # File uploader
    st.subheader("📁 Upload Audio or Video File")
    uploaded_file = st.file_uploader(
        "Choose a media file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma',  # Audio formats
              'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v', '3gp'],  # Video formats
        help="Supported formats:\n**Audio:** MP3, WAV, M4A, FLAC, OGG, WMA\n**Video:** MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP"
    )
    
    if uploaded_file is not None:
        # Determine file type
        is_video = is_video_file(uploaded_file.name)
        file_type = "Video" if is_video else "Audio"
        
        # Display file info
        st.success(f"{file_type} file uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # Media preview
        st.subheader(f"🔊 {file_type} Preview")
        if is_video:
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file)
        
        # Show video info if it's a video file
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            video_info = get_video_info(tmp_file_path)
            if video_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with col2:
                    st.metric("FPS", f"{video_info['fps']:.1f}")
                with col3:
                    st.metric("Resolution", video_info['resolution'])
            
            os.unlink(tmp_file_path)
        
        # Transcription settings
        st.subheader("⚙️ Transcription Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_length = st.slider(
                "Chunk length (seconds)",
                min_value=10,
                max_value=60,
                value=30,
                help="Length of audio chunks for processing. Shorter chunks = more precise timestamps"
            )
        
        with col2:
            generate_srt = st.checkbox("Generate SRT file", value=True)
            generate_txt = st.checkbox("Generate plain text file", value=True)
        
        # Information about SRT usage
        if is_video:
            st.info("💡 **Tip:** The generated SRT file can be loaded alongside your video in any media player (VLC, Windows Media Player, etc.) for subtitles!")
        
        # Transcribe button
        if st.button("🚀 Start Transcription", type="primary"):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load and process audio (extract from video if needed)
                with st.spinner(f"Processing {file_type.lower()} file..."):
                    audio_array, sampling_rate = load_audio_file(tmp_file_path, is_video)
                
                if audio_array is not None:
                    # Show audio info
                    duration = len(audio_array) / 16000
                    st.info(f"Audio duration: {duration:.2f} seconds")
                    
                    # Transcribe with progress bar
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        transcriptions, timestamps = transcribe_audio_with_timestamps(
                            model, processor, device, audio_array, chunk_length
                        )
                    
                    if transcriptions:
                        st.success(f"Transcription completed! Found {len(transcriptions)} segments.")
                        
                        # Display results
                        st.subheader("📝 Transcription Results")
                        
                        # Show transcriptions with timestamps
                        for i, (text, (start_time, end_time)) in enumerate(zip(transcriptions, timestamps), 1):
                            with st.expander(f"Segment {i} ({format_timestamp(start_time)} - {format_timestamp(end_time)})"):
                                st.write(text)
                        
                        # Prepare downloads
                        st.subheader("📥 Download Files")
                        
                        # Create base filename without extension
                        base_filename = Path(uploaded_file.name).stem
                        
                        col1, col2 = st.columns(2)
                        
                        if generate_srt and timestamps:
                            with col1:
                                srt_content = create_srt_content(transcriptions, timestamps)
                                st.download_button(
                                    label="📄 Download SRT file",
                                    data=srt_content.encode('utf-8'),
                                    file_name=f"{base_filename}.srt",
                                    mime="text/plain",
                                    help="Download SRT subtitle file that works with any media player"
                                )
                        
                        if generate_txt:
                            with col2:
                                txt_content = create_plain_text(transcriptions)
                                st.download_button(
                                    label="📋 Download Text file",
                                    data=txt_content.encode('utf-8'),
                                    file_name=f"{base_filename}.txt",
                                    mime="text/plain"
                                )
                        
                        # Instructions for using SRT with video
                        if is_video and generate_srt:
                            st.subheader("🎬 How to use SRT with your video")
                            st.markdown("""
                            **To add subtitles to your video:**
                            1. **Download the SRT file** using the button above
                            2. **Save both files** (video and SRT) in the same folder
                            3. **Make sure they have the same name** (e.g., `myvideo.mp4` and `myvideo.srt`)
                            4. **Open the video** in any media player (VLC, Windows Media Player, etc.)
                            5. **Subtitles will load automatically** or use the player's subtitle menu
                            
                            **Popular Media Players:**
                            - **VLC Player:** Subtitles → Add Subtitle File
                            - **Windows Media Player:** Right-click → Lyrics, Captions, and Subtitles
                            - **PotPlayer:** Right-click → Subtitle → Load Subtitle
                            - **MPC-HC:** File → Load Subtitle
                            """)
                        
                        # Full transcription display
                        st.subheader("📖 Full Transcription")
                        full_text = "\n\n".join(transcriptions)
                        st.text_area("Complete transcription:", value=full_text, height=200)
                    
                    else:
                        st.warning("No transcription generated. The audio might be too quiet or contain no speech.")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Instructions
    st.sidebar.title("📖 Instructions")
    st.sidebar.markdown("""
    1. **Upload** an audio or video file
    2. **Preview** the media using the built-in player
    3. **Adjust** settings if needed
    4. **Click** "Start Transcription" to begin
    5. **Download** SRT and/or text files
    6. **Use SRT with video** in any media player
    
    **Supported Formats:**
    
    **Audio:** MP3, WAV, M4A, FLAC, OGG, WMA
    
    **Video:** MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP
    """)
    
    st.sidebar.title("🎬 SRT Usage")
    st.sidebar.markdown("""
    **For Video Files:**
    - SRT files work with ALL major media players
    - Keep video and SRT files in same folder
    - Use same filename (video.mp4 + video.srt)
    - Most players load subtitles automatically
    
    **Compatible Players:**
    - VLC Media Player
    - Windows Media Player
    - PotPlayer
    - MPC-HC
    - KMPlayer
    - And many more!
    """)
    
    st.sidebar.title("⚠️ Requirements")
    st.sidebar.markdown("""
    **System Requirements:**
    - FFmpeg must be installed for video processing
    - Python packages: streamlit, torch, librosa, transformers
    
    **Notes:**
    - Video files: Audio is extracted using FFmpeg
    - Processing time depends on file length
    - Shorter chunks = more precise timestamps
    - Model works best with clear Nepali speech
    - Large files may take several minutes
    """)
    
    # FFmpeg status indicator
    if check_ffmpeg():
        st.sidebar.success("✅ FFmpeg detected")
    else:
        st.sidebar.error("❌ FFmpeg not found")

if __name__ == "__main__":
    main()