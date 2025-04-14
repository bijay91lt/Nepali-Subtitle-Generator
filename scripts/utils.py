import os

def print_tree(startpath, max_depth=None, prefix=""):
    def _print_tree(current_path, depth=0):
        if max_depth is not None and depth > max_depth:
            return

        entries = sorted(os.listdir(current_path))
        for idx, entry in enumerate(entries):
            path = os.path.join(current_path, entry)
            connector = "├── " if idx < len(entries) - 1 else "└── "
            print("    " * depth + connector + entry)
            if os.path.isdir(path):
                _print_tree(path, depth + 1)

    print(f"📁 {os.path.basename(startpath)}/")
    _print_tree(startpath)

# === USAGE ===
if __name__ == "__main__":
    dataset_path = "data/public_datasets/common_voice_nepali"  # Change to your dataset root
    print_tree(dataset_path, max_depth=1)  # Optional depth control
