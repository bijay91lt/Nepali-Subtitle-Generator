import os
import pandas as pd

def normalize_path(path):
    # Replace backslashes with forward slashes for cross-platform compatibility
    return path.replace("\\", "/")

def check_audio_paths(metadata_file):
    try:
        df = pd.read_csv(metadata_file)
    except Exception as e:
        print(f"Error reading the metadata file: {e}")
        return

    if 'audio' not in df.columns:
        print("The 'audio' column is missing in the metadata file.")
        return

    correct_count = 0
    incorrect_count = 0

    for index, row in df.iterrows():
        audio_path = row['audio']
        normalized_path = normalize_path(audio_path)  # Normalize the path

        if os.path.exists(normalized_path):
            correct_count += 1
        else:
            incorrect_count += 1
            print(f"Incorrect path: {normalized_path}")  # Optional: Print incorrect paths

    print(f"Correct file paths: {correct_count}")
    print(f"Incorrect file paths: {incorrect_count}")

# Example usage
metadata_file = r"data\public_datasets\merged_dataset\metadata.csv"  # Path to your metadata.csv file
check_audio_paths(metadata_file)