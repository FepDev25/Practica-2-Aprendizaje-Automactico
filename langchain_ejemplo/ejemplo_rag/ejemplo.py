"""
Ejemplo RAG (Retrieval-Augmented Generation) con LangChain y Google Vertex AI

Este script demuestra cómo crear un sistema de preguntas y respuestas
que usa documentos propios como fuente de información.

Archivos:
- config.py: Configuración del sistema
- rag_system.py: Componentes del sistema RAG
- documento_ejemplo.txt: Documento de ejemplo sobre la Misión Artemis

"""

import config
import rag_system


def main():
    """
    Función principal que orquesta todo el sistema RAG.
    """
    # 1. Configurar entorno
    PROJECT_ID, _ = config.setup_environment()
    if not PROJECT_ID:
        return

    try:
        print("\n=== Inicializando Sistema RAG ===\n")
        
        # 2. Cargar y dividir documento
        splits = rag_system.load_and_split_document(config.DOCUMENT_PATH)
        
        # 3. Crear embeddings
        embeddings = rag_system.create_embeddings(PROJECT_ID)
        
        # 4. Crear base de datos vectorial
        vectorstore = rag_system.create_vector_store(splits, embeddings)
        
        # 5. Crear retriever (buscador de documentos)
        retriever = vectorstore.as_retriever()
        
        # 6. Crear LLM
        llm = rag_system.create_llm(PROJECT_ID)
        
        # 7. Crear prompt template
        prompt = rag_system.create_prompt_template()
        
        # 8. Crear cadena RAG completa
        chain = rag_system.create_rag_chain(retriever, llm, prompt)
        
        # 9. Interfaz de usuario
        print("=== ¡Sistema RAG listo! ===")
        print("\nPreguntas sugeridas:")
        print("  • ¿Cuál es el objetivo de Artemis III?")
        print("  • ¿Qué es Artemis I?")
        print("  • ¿Cuándo está planeada Artemis II?")
        print("\nPrueba también preguntar algo fuera de contexto.")
        print("\n" + "="*50 + "\n")

        # Loop de interacción
        while True:
            pregunta = input("Tu pregunta (o 'salir' para terminar): ")
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("\nBye bye!")
                break
                
            if not pregunta.strip():
                continue
                
            print("\nBuscando en documentos y generando respuesta...\n")
            
            # Invocar la cadena RAG
            respuesta = chain.invoke(pregunta)
            
            print("Respuesta:")
            print("-" * 50)
            print(respuesta)
            print("-" * 50 + "\n")

    except Exception as e:
        print("\nError al ejecutar el sistema RAG:")
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()