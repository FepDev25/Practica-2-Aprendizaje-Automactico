from pathlib import Path

# Base del proyecto (directorio `backend`)
BASE_DIR = Path(__file__).resolve().parent

# Directorios importantes
MODEL_DIR = BASE_DIR / "model"
FILES_DIR = MODEL_DIR / "files"


def resolve_file(path_or_name: str):
    """Resuelve una ruta o nombre de archivo a una ruta absoluta dentro
    de `backend/model/files` cuando se pasa un nombre relativo.

    Si `path_or_name` ya es una ruta absoluta, la devuelve como Path.
    Si ya contiene 'model/files/', evita duplicación.
    """
    p = Path(path_or_name)
    
    # Si es absoluta, usarla directamente
    if p.is_absolute():
        return p
    
    # Si ya incluye "model/files/" al inicio, resolver desde BASE_DIR
    if path_or_name.startswith("model/files/"):
        return BASE_DIR / path_or_name
    
    # Si solo es un nombre de archivo, buscar en FILES_DIR
    candidate = FILES_DIR / path_or_name
    return candidate
