from datasets import load_dataset, Audio
import os

# --- Configuration ---
DATA_DIR = "data/public_datasets" # Relative path to your data directory
CV_DATASET_NAME = "mozilla-foundation/common_voice_11_0"
CV_LANG_CODE = "ne" # Nepali
FLEURS_DATASET_NAME = "google/fleurs"
FLEURS_LANG_CODE = "ne_np" # Nepali code in FLEURS

# --- Ensure target directory exists ---
os.makedirs(os.path.join(DATA_DIR, "common_voice_nepali"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "fleurs_nepali"), exist_ok=True)

# --- Load and potentially save Common Voice Nepali ---
print(f"Loading Common Voice dataset: {CV_DATASET_NAME} ({CV_LANG_CODE})")
try:
    # Load splits (train, validation, test, other, invalidated)
    # Set use_auth_token=True if needed (e.g., for gated datasets or newer versions)
    cv_dataset = load_dataset(CV_DATASET_NAME, CV_LANG_CODE, cache_dir=os.path.join(DATA_DIR, ".cache"),trust_remote_code=True)
    print("Common Voice dataset loaded.")

    # Optional: Save specific splits to disk in arrow format (efficient)
    # cv_dataset["train"].save_to_disk(os.path.join(DATA_DIR, "common_voice_nepali", "train"))
    # cv_dataset["validation"].save_to_disk(os.path.join(DATA_DIR, "common_voice_nepali", "validation"))
    # print("Common Voice splits saved to disk.")

    # --- Process Example (Resample and show info) ---
    # Cast the audio column to 16kHz
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16000), )
    print("\nExample from Common Voice (Train split):")
    for example in cv_dataset["train"]:
        print(example)
        break  # Print only the first example

except Exception as e:
    print(f"Error loading Common Voice: {e}")

# --- Load and potentially save FLEURS Nepali ---
print(f"\nLoading FLEURS dataset: {FLEURS_DATASET_NAME} ({FLEURS_LANG_CODE})")
try:
    # FLEURS might only have train, validation, test
    fleurs_dataset = load_dataset(FLEURS_DATASET_NAME, FLEURS_LANG_CODE, cache_dir=os.path.join(DATA_DIR, ".cache"), trust_remote_code=True)
    print("FLEURS dataset loaded.")

    # Optional: Save to disk
    # fleurs_dataset["train"].save_to_disk(os.path.join(DATA_DIR, "fleurs_nepali", "train"))
    # print("FLEURS splits saved to disk.")

    # --- Process Example (Already 16kHz usually) ---
    print("\nExample from FLEURS (Train split):")
    # Access features correctly (might differ slightly based on dataset version)
    example = fleurs_dataset["train"][0]
    print(f"Path: {example.get('path', 'N/A')}")
    print(f"Audio sampling rate: {example['audio']['sampling_rate']}")
    print(f"Raw Transcription: {example.get('raw_transcription', 'N/A')}")
    print(f"Transcription: {example.get('transcription', example.get('raw_transcription'))}") # Use best available text


except Exception as e:
    print(f"Error loading FLEURS: {e}")

print("\nData acquisition script finished.")