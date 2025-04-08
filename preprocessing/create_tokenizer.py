import os
from transformers import Wav2Vec2CTCTokenizer

VOCAB_PATH = "data/vocab.json"
TOKENIZER_DIR = "data/tokenizer"

def create_tokenizer():
    print("Creating Wav2Vec2CTCTokenizer...")
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=VOCAB_PATH,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    print(f"Saving tokenizer to: {TOKENIZER_DIR}")
    tokenizer.save_pretrained(TOKENIZER_DIR)

if __name__ == "__main__":
    create_tokenizer()
    print("Tokenizer created and saved successfully.")

