from fastapi import FastAPI
from model.registro import Registros
from db import session
import cv2

app = FastAPI()

@app.post("/")
async def insert_Data(txt1:str,txt2:str):
    print(cv2.__version__)
    registro = Registros(text1=txt1,text2=txt2)
    session.add(registro)
    session.commit()
    
    return{"resp": registro.id}
