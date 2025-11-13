from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

from db import engine

Base = declarative_base()

class Registros(Base):  
    __tablename__ = "registros"  
    id = Column(Integer, primary_key=True)
    text1 = Column(String)
    text2 = Column(String)
    

Base.metadata.create_all(engine)