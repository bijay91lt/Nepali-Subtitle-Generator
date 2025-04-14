import os
from pydub import AudioSegment

# Path to the directory containing your .mp3 files
input_dir = r"data\public_datasets\common_voice_nepali\clips"  # Replace this with your actual path
output_dir = r"data\public_datasets\common_voice_nepali\wav_clips"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Walk through the directory tree
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".mp3"):
            mp3_path = os.path.join(root, file)
            
            # Define the output .wav path (preserving folder structure)
            relative_path = os.path.relpath(root, input_dir)
            wav_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(wav_subdir, exist_ok=True)

            wav_path = os.path.join(wav_subdir, os.path.splitext(file)[0] + ".wav")

            # Load and export
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")

            print(f"Converted: {mp3_path} → {wav_path}")
