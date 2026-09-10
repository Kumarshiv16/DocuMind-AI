"""
Utility script to create a clean, distribution-ready DocuMind_AI_Web_Project.zip
Excludes virtual environments, cache files, and private .env secrets.
"""

import os
import zipfile

OUTPUT_ZIP = "DocuMind_AI_Web_Project.zip"

EXCLUDE_DIRS = {
    "venv", "env", "ENV", ".git", ".idea", ".vscode", "__pycache__", "vector_db", ".streamlit"
}

EXCLUDE_FILES = {
    ".env", OUTPUT_ZIP, ".DS_Store", "Thumbs.db"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".log"
}


def make_zip():
    project_root = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_root, OUTPUT_ZIP)

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_root):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, project_root)
                zf.write(full_path, rel_path)
                file_count += 1

    file_size_kb = round(os.path.getsize(zip_path) / 1024, 1)
    print(f"Created {OUTPUT_ZIP} ({file_size_kb} KB) containing {file_count} project files.")


if __name__ == "__main__":
    make_zip()
