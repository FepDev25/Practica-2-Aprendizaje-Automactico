from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
FILES_DIR = MODEL_DIR / "files"


def resolve_file(path_or_name: str):
    p = Path(path_or_name)
    
    if p.is_absolute():
        return p
    
    if path_or_name.startswith("model/files/"):
        return BASE_DIR / path_or_name
    
    candidate = FILES_DIR / path_or_name
    return candidate
