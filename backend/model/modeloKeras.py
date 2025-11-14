from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from io import StringIO
import sys

class ModeloKeras:
    def __init__(self, pesos_path='best_model.keras'):
        # Definición del modelo (ajústala según tu arquitectura)
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(1, activation='sigmoid')
        ])
        self.pesos_path = pesos_path
        self._cargar_pesos()

    def _cargar_pesos(self):
        try:
            self.model.load_weights(self.pesos_path)
            print(f"Pesos cargados desde {self.pesos_path}")
        except Exception as e:
            print(f"Error al cargar los pesos: {e}")

    def obtener_resumen(self) -> str:
        stream = StringIO()
        sys.stdout = stream
        self.model.summary()
        sys.stdout = sys.__stdout__
        return stream.getvalue()
    
    def predecir(self, features):
        
        if self.model is None:
            raise ValueError("El modelo no está cargado o no existe.")

        
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)
        return float(prediction[0][0])  
