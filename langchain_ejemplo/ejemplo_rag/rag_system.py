from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_and_split_document(file_path, chunk_size=None, chunk_overlap=None):
    """
    Carga un documento y lo divide en fragmentos (chunks).
    
    Args:
        file_path: Ruta al archivo de texto
        chunk_size: Tamaño de cada fragmento
        chunk_overlap: Superposición entre fragmentos
        
    Returns:
        list: Lista de documentos divididos
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    # Cargar documento
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    
    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"Documento cargado y dividido en {len(splits)} fragmentos")
    return splits


def create_embeddings(project_id):
    """
    Crea el modelo de embeddings para convertir texto a vectores.
    
    Args:
        project_id: ID del proyecto de Google Cloud
        
    Returns:
        VertexAIEmbeddings: Modelo de embeddings
    """
    embeddings = VertexAIEmbeddings(
        model=config.EMBEDDING_MODEL, 
        project=project_id
    )
    print(f"Modelo de embeddings '{config.EMBEDDING_MODEL}' inicializado")
    return embeddings


def create_vector_store(documents, embeddings):
    """
    Crea una base de datos vectorial en memoria con FAISS.
    
    Args:
        documents: Lista de documentos divididos
        embeddings: Modelo de embeddings
        
    Returns:
        FAISS: Base de datos vectorial
    """
    print("Creando base de datos vectorial (FAISS)...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Base de datos vectorial creada")
    return vectorstore


def create_llm(project_id):
    """
    Crea el modelo de lenguaje (LLM).
    
    Args:
        project_id: ID del proyecto de Google Cloud
        
    Returns:
        ChatVertexAI: Modelo de lenguaje
    """
    llm = ChatVertexAI(
        model=config.LLM_MODEL, 
        project=project_id,
    )
    print(f"Modelo LLM '{config.LLM_MODEL}' inicializado")
    return llm


def create_prompt_template():
    """
    Crea la plantilla de prompt para el RAG.
    Esta plantilla define cómo el LLM debe usar el contexto recuperado.
    
    Returns:
        ChatPromptTemplate: Plantilla de prompt
    """
    template = """
    Eres un asistente experto en misiones espaciales.
    Usa ÚNICAMENTE el siguiente contexto para responder la pregunta.
    Si no sabes la respuesta o el contexto no la contiene, di "No tengo información sobre eso".

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """
    return ChatPromptTemplate.from_template(template)


def create_rag_chain(retriever, llm, prompt):
    """
    Crea la cadena RAG completa usando LCEL (LangChain Expression Language).
    
    Args:
        retriever: Recuperador de documentos
        llm: Modelo de lenguaje
        prompt: Plantilla de prompt
        
    Returns:
        Chain: Cadena RAG completa
    """
    # 1. retriever busca documentos relevantes (se asignan a 'context')
    # 2. RunnablePassthrough() pasa la pregunta sin modificar (se asigna a 'question')
    # 3. prompt inserta context y question en la plantilla
    # 4. llm genera la respuesta basada en el prompt
    # 5. StrOutputParser() convierte la respuesta a string limpio
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Cadena RAG creada y lista para usar\n")
    return chain
