import os
from langchain_google_vertexai import ChatVertexAI

def main():
    PROJECT_ID = os.getenv("PROJECT_ID")
    CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not PROJECT_ID:
        print("Error: La variable de entorno PROJECT_ID no está configurada.")
        return
        
    if not CREDENTIALS_PATH:
        print("Error: La variable GOOGLE_APPLICATION_CREDENTIALS no está configurada.")
        return

    print(f"--- Configuración ---")
    print(f"Proyecto: {PROJECT_ID}")
    print(f"Credenciales: {CREDENTIALS_PATH}")
    print("----------------------\n")

    try:
        llm = ChatVertexAI(
            model="gemini-2.5-pro", 
            project=PROJECT_ID,
        )

        print("Conectando con Vertex AI (Gemini)...")

        prompt = input("Escribe tu pregunta para el modelo Gemini: ")
        print("\nEnviando la pregunta al modelo...")

        message = llm.invoke(prompt) # Realiza la llamada al modelo
        
        print("\n--- Respuesta del Modelo ---")
        print(message.content)
        print("--------------------------")

    except Exception as e:
        print("\n--- ¡Error! ---")
        print(f"Ha ocurrido un error al conectar con Vertex AI:")
        print(e)

if __name__ == "__main__":
    main()