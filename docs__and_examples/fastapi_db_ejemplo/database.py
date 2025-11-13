"""
Configuración de la base de datos PostgreSQL
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/ejemplo1"

# Crear el motor de base de datos
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Muestra las consultas SQL en consola (útil para debugging)
    pool_pre_ping=True,  # Verifica la conexión antes de usar
    pool_size=5,  # Número de conexiones en el pool
    max_overflow=10  # Conexiones adicionales permitidas
)

# Crear una sesión local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para los modelos
Base = declarative_base()

# Dependencia para obtener la sesión de base de datos
def get_db():
    """
    Generador que proporciona una sesión de base de datos
    y asegura que se cierre después de usar
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
