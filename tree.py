import os
import argparse
from pathlib import Path

def generate_tree(dir_path: Path, prefix: str = "", ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = {
            '.git', '__pycache__', 'venv', 'env', '.venv', '.idea', 
            '.vscode', 'node_modules', '.ipynb_checkpoints', '.DS_Store'
        }
        
    try:
        # Get all entries in the directory, sorted (directories first, then files)
        entries = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        # Filter out ignored directories and files
        entries = [e for e in entries if e.name not in ignore_dirs]
        
        for i, entry in enumerate(entries):
            is_last = i == (len(entries) - 1)
            connector = "└── " if is_last else "├── "
            
            print(prefix + connector + entry.name)
            
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                generate_tree(entry, prefix + extension, ignore_dirs)
                
    except PermissionError:
        print(prefix + "└── [Permission Denied]")

def main():
    parser = argparse.ArgumentParser(description="Generate a directory tree structure.")
    parser.add_argument("directory", nargs="?", default=".", help="The directory to process (default: current directory)")
    args = parser.parse_args()
    
    target_dir = Path(args.directory).resolve()
    
    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        return
        
    print(target_dir.name + "/")
    generate_tree(target_dir)

if __name__ == "__main__":
    main()
