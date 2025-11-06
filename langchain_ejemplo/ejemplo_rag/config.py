import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
env_path = Path(__file__).parent / 'env/.env'
load_dotenv(dotenv_path=env_path)

def setup_environment():
    PROJECT_ID = os.getenv("PROJECT_ID")
    CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not PROJECT_ID or not CREDENTIALS_PATH:
        print("Error: Variables de entorno no configuradas.")
        return None, None

    if not Path(CREDENTIALS_PATH).exists():
        print(f"Error: No se encontró el archivo de credenciales en:")
        print(f"   {CREDENTIALS_PATH}")
        return None, None

    print(f"--- Configuración ---")
    print(f"Proyecto: {PROJECT_ID}")
    print(f"Credenciales: {CREDENTIALS_PATH}")
    print("----------------------\n")
    
    return PROJECT_ID, CREDENTIALS_PATH


LLM_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DOCUMENT_PATH = "documento_ejemplo.txt"
