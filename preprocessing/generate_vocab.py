import json
import os
from datasets import load_dataset

DATA_DIR = "data"
TRAIN_TSV = os.path.join(DATA_DIR, "train.tsv")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": vocab}

def build_vocab():
    print("Loading training dataset...")
    ds = load_dataset("csv", data_files=TRAIN_TSV, delimiter="\t")["train"]
    
    print("Extracting unique characters...")
    vocab_set = set()
    for example in ds:
        vocab_set.update(list(example["sentence"]))

    vocab_list = sorted(list(vocab_set))
    
    # Create vocab dictionary
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}

    # Add special tokens
    vocab_dict["|"] = len(vocab_dict)            # For space
    vocab_dict["[UNK]"] = len(vocab_dict) + 1    # Unknown token
    vocab_dict["[PAD]"] = len(vocab_dict) + 2    # Padding

    print(f"Saving vocab with {len(vocab_dict)} tokens to {VOCAB_PATH}")
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    build_vocab()


# This will scan all text in train.tsv

# Space (" ") is replaced by "|" as Wav2Vec2 expects this

# Special tokens are added at the end