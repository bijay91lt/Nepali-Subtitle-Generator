import os
import pandas as pd

# Path to your dataset folder
dataset_dir = r"data\public_datasets\common_voice_nepali"  # e.g., r"C:\Users\Kakeru\Downloads\dataset"
old_subfolder = "clips"
new_subfolder = "wav_clips"
old_ext = ".mp3"
new_ext = ".wav"

# Loop through all TSV files in the directory
for filename in os.listdir(dataset_dir):
    if filename.endswith(".tsv"):
        file_path = os.path.join(dataset_dir, filename)

        df = pd.read_csv(file_path, sep='\t', dtype=str)

        # If there's a 'path' column (usually in Common Voice), update it
        if 'path' in df.columns:
            df['path'] = df['path'].apply(lambda p: p.replace(old_subfolder, new_subfolder).replace(old_ext, new_ext))

        # Save back to file (backup optional)
        df.to_csv(file_path, sep='\t', index=False)
        print(f"✅ Updated paths in: {filename}")
