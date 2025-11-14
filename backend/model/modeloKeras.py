import joblib
import numpy as np
import pandas as pd
from io import StringIO
import sys


class ModeloStockRF:
    def __init__(self, modelo_path="modelo_stock_rfr.joblib"):
        """
        modelo_path: ruta donde guardaste tu modelo entrenado.
        """
        self.modelo_path = modelo_path
        self.model = None
        self._cargar_modelo()


    def _cargar_modelo(self):
        try:
            self.model = joblib.load(self.modelo_path)
            print(f"Modelo cargado desde: {self.modelo_path}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None


    def obtener_resumen(self) -> str:
        
        if self.model is None:
            return "Modelo no cargado."

        stream = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = stream

        print("==== RESUMEN DEL MODELO RANDOM FOREST ====")
        print(f"Tipo de modelo: {type(self.model).__name__}")
        print(f"Número de árboles: {self.model.n_estimators}")
        print(f"Máxima profundidad: {self.model.max_depth}")
        print(f"Features usados: {self.model.n_features_in_}")

        sys.stdout = sys_stdout
        return stream.getvalue()


    def predecir(self, features: dict):
        """
        Recibe un dict con los valores de los features.
        Devuelve la predicción como float.
        """

        if self.model is None:
            raise ValueError("El modelo no está cargado o no existe.")

        # Convertir el dict en DataFrame (igual que en el entrenamiento)
        X_input = pd.DataFrame([features])

        pred = self.model.predict(X_input)
        return float(pred[0])
