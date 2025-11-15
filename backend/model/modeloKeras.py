import numpy as np
import pandas as pd
from io import StringIO
import sys
from tensorflow.keras.models import load_model
from joblib import load
    
scaler_y = load("scaler_y.joblib")
class ModeloStockKeras:
    def __init__(self, modelo_path="best_model.keras"):
        """
        modelo_path: ruta donde guardaste tu modelo Keras .
        """
        self.modelo_path = modelo_path
        self.model = None
        self._cargar_modelo()

    def _cargar_modelo(self):
        try:
            self.model = load_model(self.modelo_path)
            print(f"Modelo Keras cargado desde: {self.modelo_path}")
        except Exception as e:
            print(f"Error al cargar el modelo Keras: {e}")
            self.model = None

    def obtener_resumen(self) -> str:
        """
        Devuelve el summary() del modelo como string.
        """
        if self.model is None:
            return "Modelo Keras no cargado."

        stream = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = stream

        print("==== RESUMEN DEL MODELO KERAS ====")
        self.model.summary()

        sys.stdout = sys_stdout
        return stream.getvalue()

    def predecir(self, X_input):
        """
        Recibe un dict con valores numéricos.
        Convierte a DataFrame y luego a numpy para hacer predict().
        """

        if self.model is None:
            raise ValueError("El modelo Keras no está cargado.")

        # Predecir
        pred = self.model.predict(X_input, verbose=0)
        pred_real = scaler_y.inverse_transform(pred)
        # Si el modelo solo tiene 1 output
        return float(pred_real[0][0])