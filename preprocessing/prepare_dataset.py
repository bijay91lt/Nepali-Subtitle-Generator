import os
from datasets import load_dataset, DatasetDict, Audio

DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "clips")

def load_commonvoice_tsv(split):
    path = os.path.join(DATA_DIR, f"{split}.tsv")
    return load_dataset("csv", data_files=path, delimiter="\t")["train"]

def add_audio_path(example):
    example["audio"] = os.path.join(AUDIO_DIR, example["path"])
    return example

def prepare_dataset():
    data = DatasetDict()
    for split in ["train", "test", "dev"]:
        print(f"Loading {split}...")
        ds = load_commonvoice_tsv(split)
        ds = ds.map(add_audio_path)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = ds.rename_column("sentence", "text")  # Rename to match HuggingFace conventions
        data[split] = ds
    return data

if __name__ == "__main__":
    dataset = prepare_dataset()
    print(dataset)
    print(dataset["train"][0])
