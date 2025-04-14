# scripts/1_prepare_data.py

import os
import re
# Make sure to import necessary libraries
from datasets import load_dataset, DatasetDict, Audio, Value
import pandas as pd # Optional: Useful for inspecting the CSV first

# --- Configuration ---
# ---vvv--- ADJUST THESE ---vvv---
CSV_METADATA_PATH = r"data\public_datasets\merged_dataset\metadata.csv"  # <--- CHANGE THIS to the path of your CSV file
AUDIO_FOLDER_PATH = r"data\public_datasets\merged_dataset\audio"      # <--- Set the *base* path to your audio files if paths in CSV are relative to this folder. If paths in CSV are absolute, this might not be strictly needed but helps organize.
AUDIO_PATH_COLUMN = "audio_path"                # <--- CHANGE THIS to the exact name of the column in your CSV with audio file paths
TRANSCRIPTION_COLUMN = "transcription"          # <--- CHANGE THIS to the exact name of the column in your CSV with transcriptions
# ---^^^--- ADJUST THESE ---^^^---

PROCESSED_DATA_PATH = "../data/processed_nepali_whisper"
MODEL_ID = "openai/whisper-small"
TARGET_LANGUAGE = "ne"
TASK = "transcribe"
TEST_SPLIT_SIZE = 0.10  # Percentage of data for the test set (e.g., 10%)
VALIDATION_SPLIT_SIZE = 0.10 # Percentage of the *remaining* data for validation (e.g., 10% of the original 90%)

# --- Optional: Inspect your CSV (good practice) ---
try:
    print(f"Inspecting CSV file: {CSV_METADATA_PATH}")
    df = pd.read_csv(CSV_METADATA_PATH)
    print("CSV Head:")
    print(df.head())
    print(f"\nColumns found: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    # Check if assumed columns exist
    if AUDIO_PATH_COLUMN not in df.columns:
        print(f"WARNING: Assumed audio path column '{AUDIO_PATH_COLUMN}' not found in CSV!")
    if TRANSCRIPTION_COLUMN not in df.columns:
        print(f"WARNING: Assumed transcription column '{TRANSCRIPTION_COLUMN}' not found in CSV!")
except Exception as e:
    print(f"Could not inspect CSV with pandas: {e}")


# --- Load the Dataset from CSV ---
print(f"Loading dataset from CSV: {CSV_METADATA_PATH}")
try:
    # Load the entire dataset from the CSV file
    # The 'datasets' library often handles paths relative to the CSV location well.
    # If audio paths are absolute, it should also work.
    # If paths are relative to a different base (AUDIO_FOLDER_PATH), casting the audio column later handles loading.
    full_dataset = load_dataset("csv", data_files=CSV_METADATA_PATH, split="train") # Load as 'train' split initially
    print(f"Full dataset loaded successfully. Number of samples: {len(full_dataset)}")

    # --- Important: Rename columns to expected names if they differ ---
    # If your CSV columns aren't AUDIO_PATH_COLUMN and TRANSCRIPTION_COLUMN, rename them now
    # This makes subsequent steps cleaner. Example:
    # full_dataset = full_dataset.rename_column("your_file_column", AUDIO_PATH_COLUMN)
    # full_dataset = full_dataset.rename_column("your_text_column", TRANSCRIPTION_COLUMN)

except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_METADATA_PATH}. Make sure the path is correct.")
    exit()
except Exception as e:
    print(f"Error loading dataset from CSV: {e}")
    exit()

# --- Split the Dataset ---
print("Splitting the dataset into train, validation, and test sets...")

# 1. Split into train+validation and test
train_validation_split = full_dataset.train_test_split(test_size=TEST_SPLIT_SIZE, shuffle=True, seed=42) # Use seed for reproducibility
test_dataset = train_validation_split["test"]
train_validation_dataset = train_validation_split["train"]

# 2. Split train+validation into train and validation
# Calculate split size relative to the *remaining* data
relative_validation_size = VALIDATION_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
train_val_split = train_validation_dataset.train_test_split(test_size=relative_validation_size, shuffle=True, seed=42)
train_dataset = train_val_split["train"]
validation_dataset = train_val_split["test"]

# 3. Combine into a DatasetDict
merged_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

print("Dataset split complete:")
print(merged_datasets)

# --- Load Whisper Processor ---
from transformers import WhisperProcessor
print(f"Loading Whisper processor for: {MODEL_ID}")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=TARGET_LANGUAGE, task=TASK)


# --- Crucial Step: Cast Audio Column ---
# This tells the 'datasets' library to treat the string path in the AUDIO_PATH_COLUMN
# as an audio file, load it, and resample it to 16kHz.
# It needs the base path if the paths in the CSV are relative.
print(f"Casting audio column '{AUDIO_PATH_COLUMN}' to Audio feature (16kHz)...")
# Ensure the column containing the path exists BEFORE casting
if AUDIO_PATH_COLUMN not in merged_datasets["train"].column_names:
     print(f"ERROR: Column '{AUDIO_PATH_COLUMN}' not found in the dataset columns: {merged_datasets['train'].column_names}")
     print("Please check the AUDIO_PATH_COLUMN variable and your CSV file.")
     exit()

try:
    # The key is mapping the column with file paths (AUDIO_PATH_COLUMN) to the Audio feature.
    # It automatically loads and decodes the audio file found at the path.
    merged_datasets = merged_datasets.cast_column(AUDIO_PATH_COLUMN, Audio(sampling_rate=16000))
    # --- Rename the audio column to 'audio' for consistency ---
    # The rest of the script might expect the audio data to be in a column named 'audio'
    merged_datasets = merged_datasets.rename_column(AUDIO_PATH_COLUMN, "audio")
    print("Audio casting and renaming complete.")
    print("Sample audio info after casting:")
    print(merged_datasets["train"][0]["audio"]) # Now access via 'audio'
except Exception as e:
    print(f"Error casting audio column '{AUDIO_PATH_COLUMN}': {e}")
    print("Check that the paths in your CSV are correct relative to the CSV file or absolute.")
    print("Also ensure necessary audio libraries like 'libsndfile' or 'ffmpeg' are installed.")
    exit()


# --- Text Normalization/Cleaning Function (Customize for Nepali) ---
# (Keep the function definition as before)
def normalize_nepali_text(text):
    if not isinstance(text, str):
        return ""
    # Add your Nepali-specific cleaning rules here (using re.sub, etc.)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Preprocessing Function using the Processor ---
# (Make sure the transcription column name is correct here)
def prepare_dataset(batch):
    # Audio Processing: Extract features from the 'audio' column (already loaded and resampled)
    try:
        # Ensure 'audio' column data is accessed correctly (list of dicts {'array': ..., 'sampling_rate': ...})
        audio_input = [item["array"] for item in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"] # Get sampling rate from first item (should be 16k)

        # --- vvv CORRECTED LINE vvv ---
        # Assign the list of features directly
        batch["input_features"] = processor(audio_input, sampling_rate=sampling_rate).input_features
        # --- ^^^ CORRECTED LINE ^^^ ---

    except Exception as e:
        print(f"Error processing audio batch: {e}")
        # Ensure fallback maintains batch size structure
        batch["input_features"] = [None] * len(batch["audio"]) # List of Nones, length = batch size

    # Text Processing: Normalize and tokenize from the TRANSCRIPTION_COLUMN
    try:
        normalized_transcriptions = [normalize_nepali_text(trans) for trans in batch[TRANSCRIPTION_COLUMN]]
        batch["labels"] = processor.tokenizer(normalized_transcriptions).input_ids
    except Exception as e:
        print(f"Error processing text batch: {e}")
         # Ensure fallback maintains batch size structure
        batch["labels"] = [None] * len(batch[TRANSCRIPTION_COLUMN]) # List of Nones, length = batch size


    # --- Optional: Add a check for length consistency ---
    if len(batch["input_features"]) != len(batch["labels"]):
         print(f"WARNING: Length mismatch detected BEFORE returning batch!")
         print(f"Input features length: {len(batch['input_features'])}")
         print(f"Labels length: {len(batch['labels'])}")
         # Consider how to handle this, e.g., skip the batch or use fallback Nones for both

    return batch

# --- Apply Preprocessing ---
print("Applying preprocessing to the dataset...")

# Define columns to remove: Keep only processed columns ('input_features', 'labels')
# Get all original column names EXCEPT the ones we need to process or are keeping implicitly.
original_columns = merged_datasets["train"].column_names
columns_to_remove = [col for col in original_columns if col not in ["audio", TRANSCRIPTION_COLUMN]] # Keep audio and text cols for prepare_dataset
# Add the original audio and text columns after prepare_dataset runs, if needed for removal
final_columns_to_remove = list(set(columns_to_remove + ["audio", TRANSCRIPTION_COLUMN]))


processed_datasets = merged_datasets.map(
    prepare_dataset,
    remove_columns=final_columns_to_remove, # Remove original audio path, transcription, and any other columns
    batched=True,
    batch_size=16
    # num_proc=4 # Use multiple cores if available
).with_format("torch") # Keep as torch tensors for PyTorch

print("Preprocessing complete.")
print("Processed dataset columns:", processed_datasets["train"].column_names) # Should only be input_features, labels

# --- Save Processed Data ---
# (Save logic remains the same as before)
print(f"Saving processed dataset to: {PROCESSED_DATA_PATH}")
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
processed_datasets.save_to_disk(PROCESSED_DATA_PATH)

print("Processed dataset saved successfully!")
print(f"Dataset splits created and processed. Ready for training.")