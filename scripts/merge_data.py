import csv
import os
from pathlib import Path
from pydub import AudioSegment

# Increase the field size limit to handle large fields
csv.field_size_limit(1000000)  # Set a large but safe limit

# Define paths
ROOT = Path("data/public_datasets")
MERGED_AUDIO_DIR = ROOT / "merged_dataset/audio"
MERGED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MERGED_CSV_PATH = ROOT / "merged_dataset/metadata.csv"

def convert_to_wav(src_path, dst_path):
    """Convert audio file to WAV format with 16kHz sample rate and mono channel."""
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(dst_path, format="wav")
    except Exception as e:
        print(f"Failed to convert {src_path} to {dst_path}. Error: {e}")

def process_common_voice(folder, records):
    """Process Common Voice Nepali dataset."""
    tsv_files = ["train.tsv", "test.tsv", "dev.tsv"]
    wav_dir = folder / "wav_clips"
    
    for tsv_file in tsv_files:
        tsv_path = folder / tsv_file
        if not tsv_path.exists():
            print(f"Skipping missing file: {tsv_path}")
            continue

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mp3_name = row["path"]
                transcript = row["sentence"]
                src = wav_dir / mp3_name.replace(".mp3", ".wav")

                if not src.exists():
                    print(f"Skipping missing file: {src}")
                    continue

                dst = MERGED_AUDIO_DIR / src.name
                convert_to_wav(src, dst)
                
                # Add relative path for audio
                relative_audio_path = dst.relative_to(ROOT)
                records.append([str(relative_audio_path), transcript])

def process_fleurs(folder, records):
    """Process FLEURS Nepali dataset."""
    splits = ["train", "dev", "test"]
    for split in splits:
        tsv_path = folder / f"{split}.tsv"
        audio_dir = folder / split

        if not tsv_path.exists() or not audio_dir.exists():
            print(f"Skipping missing file or directory: {tsv_path} or {audio_dir}")
            continue

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Extract only the audio path (column 2) and transcription (column 3)
                clip_name = row[0]  # Column 2: Audio file name
                transcript = row[1]  # Column 3: Transcription
                
                # Construct the full path to the audio file
                src = audio_dir / clip_name

                # Skip if the audio file is missing
                if not src.exists():
                    print(f"Skipping missing file: {src}")
                    continue

                # Convert the audio file to WAV format and save it in the destination directory
                dst = MERGED_AUDIO_DIR / src.name
                convert_to_wav(src, dst)

                # Add the relative path for the audio file and its transcription to the records
                relative_audio_path = dst.relative_to(ROOT)
                records.append([str(relative_audio_path), transcript])

def process_openslr(folder, records):
    """Process OpenSLR Nepali dataset."""
    for subdir in folder.glob("asr_nepali*"):  # Look for subdirectories matching the pattern
        # Locate the utt_spk_text.tsv file in the asr_nepali* folder
        tsv_path = subdir / "utt_spk_text.tsv"
        if not tsv_path.exists():
            print(f"Skipping {subdir}: 'utt_spk_text.tsv' not found.")
            continue

        data_dir = subdir / "data"
        if not data_dir.exists():
            print(f"Skipping {subdir}: 'data' directory not found.")
            continue

        # Extract the prefix from the folder name (e.g., "asr_nepali_1" -> "1", "asr_nepali_a" -> "a")
        folder_prefix = subdir.name.split("_")[-1] if "_" in subdir.name else subdir.name.split("asr_nepali_")[-1]

        # Read the metadata from utt_spk_text.tsv
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:  # Ensure the row has at least 3 columns
                    print(f"Skipping malformed row: {row}")
                    continue

                clip_name = row[0] + ".flac"
                transcript = row[2]

                # Search for the .flac file in subfolders matching the folder_prefix
                found = False
                for subfolder in data_dir.iterdir():
                    if not subfolder.is_dir():
                        continue

                    # Check if the subfolder starts with the folder_prefix
                    if subfolder.name.startswith(folder_prefix):
                        src = subfolder / clip_name
                        if src.exists():
                            dst = MERGED_AUDIO_DIR / (row[0] + ".wav")
                            convert_to_wav(src, dst)
                            
                            # Add relative path for audio
                            relative_audio_path = dst.relative_to(ROOT)
                            records.append([str(relative_audio_path), transcript])
                            found = True
                            break

                if not found:
                    # Debugging: Check if the file exists in any subfolder
                    possible_locations = []
                    for subfolder in data_dir.iterdir():
                        if subfolder.is_dir() and subfolder.name.startswith(folder_prefix):
                            possible_locations.append(subfolder / clip_name)

                    print(f"Skipping missing file: {clip_name}")
                    print(f"Checked locations: {[str(loc) for loc in possible_locations]}")

def merge_all():
    """Merge all datasets into the unified dataset."""
    records = []
    process_common_voice(ROOT / "common_voice_nepali", records)
    process_fleurs(ROOT / "fleurs_nepali", records)
    # process_openslr(ROOT / "openslr_nepali", records)

    # Write metadata to CSV
    with open(MERGED_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "sentence"])
        writer.writerows(records)
    print(f"Merged {len(records)} audio-transcription pairs into {MERGED_CSV_PATH}")

if __name__ == "__main__":
    merge_all()
