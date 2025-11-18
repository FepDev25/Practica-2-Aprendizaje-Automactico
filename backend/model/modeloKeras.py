import numpy as np
import pandas as pd
from io import StringIO
import sys
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import pandas as pd
from joblib import load, dump
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from paths import resolve_file, FILES_DIR



N_STEPS = 7
# Cargar scaler_y desde ubicación absoluta (production-safe)
scaler_y = load(str(resolve_file("scaler_y.joblib")))

class ModeloStockKeras:
    def __init__(self, modelo_path="model/files/model.keras"):
        """
        modelo_path: ruta donde guardaste tu modelo Keras .
        """
        self.modelo_path = modelo_path
        self.model = None
        self._cargar_modelo()

    def _cargar_modelo(self):
        try:
            modelo_ruta = resolve_file(self.modelo_path)
            self.model = load_model(str(modelo_ruta))
            print(f"Modelo Keras cargado desde: {modelo_ruta}")
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
    

def create_sequences(X_data, y_data, time_steps=7):
    """Crea secuencias exactamente como en Fase_02."""
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)


def reentrenar_modelo_con_diferencias(path_original="dataset_processed_advanced.csv",
                                      path_nuevo="dataset_processed_advanced2.csv",
                                      ruta_modelo="model.keras"):
    """
    1. Carga los dos datasets.
    2. Igual las columnas entre ambos.
    3. Detecta diferencias.
    4. Reentrena el modelo GRU usando esas diferencias.
    """

    print("=== REENTRENAMIENTO AUTOMÁTICO INICIADO ===")

    # ------------------------------------------------------------------
    # 1. Cargar datasets
    # ------------------------------------------------------------------
    # Resolver rutas relativas dentro de FILES_DIR si se entregaron nombres
    path1 = resolve_file(path_original)
    path2 = resolve_file(path_nuevo)

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    print("Dataset original:", df1.shape)
    print("Dataset nuevo:", df2.shape)

    # ------------------------------------------------------------------
    # 2. Igualar columnas entre ambos datasets
    # ------------------------------------------------------------------
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = None

    for col in df2.columns:
        if col not in df1.columns:
            df1[col] = None

    # ------------------------------------------------------------------
    # 3. Ordenar columnas iguales para comparar fila por fila
    # ------------------------------------------------------------------
    df1 = df1[df2.columns]

    # ------------------------------------------------------------------
    # 4. Resetear índices
    # ------------------------------------------------------------------
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Obtener solo filas distintas
    # ------------------------------------------------------------------
    mask_diferencias = df1.ne(df2).any(axis=1)
    diferencias = df1[mask_diferencias]

    print("Filas diferentes encontradas:", diferencias.shape)

    if diferencias.empty:
        print("No hay diferencias → no es necesario reentrenar.")
        return {"mensaje": "Sin cambios, sin reentrenamiento."}

    df_new = diferencias.copy()
    
    
    try:
        out_path = resolve_file("dataset_processed_advanced.csv")
        df2.to_csv(out_path, index=False)
        print(f"df2 guardado con éxito en: {out_path}")
    except Exception as e:
        print(f"Error al guardar df2: {e}")
    # ------------------------------------------------------------------
    # 6. Eliminar columnas no numéricas como en Fase_02
    # ------------------------------------------------------------------
    cols_a_excluir = ["product_sku"]
    if "region_almacen" in df_new.columns:
        cols_a_excluir.append("region_almacen")

    df_new = df_new.drop(columns=cols_a_excluir, errors="ignore")

    # ------------------------------------------------------------------
    # 7. Separar X e y
    # ------------------------------------------------------------------
    if "quantity_available" not in df_new.columns:
        raise Exception("ERROR: 'quantity_available' no está en el dataset.")

    y_new = df_new[['quantity_available']]
    X_new = df_new.drop(columns=['quantity_available'])

    # ------------------------------------------------------------------
    # 8. Cargar scalers originales
    # ------------------------------------------------------------------
    scaler_X = load(str(resolve_file("scaler_X.joblib")))
    scaler_y = load(str(resolve_file("scaler_y.joblib")))

    print("Scalers cargados.")

    # ------------------------------------------------------------------
    # 9. Escalar igual que el entrenamiento original
    # ------------------------------------------------------------------
    X_new_scaled = scaler_X.transform(X_new)
    y_new_scaled = scaler_y.transform(y_new)

    # ------------------------------------------------------------------
    # 10. Crear secuencias de 7 días
    # ------------------------------------------------------------------
    X_seq, y_seq = create_sequences(X_new_scaled, y_new_scaled, N_STEPS)

    print("Secuencias creadas:")
    print("X_seq:", X_seq.shape)
    print("y_seq:", y_seq.shape)

    if len(X_seq) < 5:
        raise Exception("Muy pocos datos para reentrenar (mín. 8 filas).")

    # ------------------------------------------------------------------
    # 11. Cargar el modelo actual
    # ------------------------------------------------------------------
    # Cargar el modelo actual (resolviendo ruta relativa si aplica)
    ruta_modelo_resuelta = resolve_file(ruta_modelo)
    model = load_model(str(ruta_modelo_resuelta))
    print("Modelo GRU cargado.")

    # ------------------------------------------------------------------
    # 12. Callbacks (igual que Fase_02)
    # ------------------------------------------------------------------
    checkpoint = ModelCheckpoint(
        filepath=ruta_modelo,
        monitor="loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        mode="min",
        verbose=1
    )

    # ------------------------------------------------------------------
    # 13. Compilar modelo igual que en Fase_02
    # ------------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    # ------------------------------------------------------------------
    # 14. Reentrenamiento (fine-tuning)
    # ------------------------------------------------------------------
    print("Reentrenando el modelo con diferencias...")

    history = model.fit(
        X_seq,
        y_seq,
        epochs=1,
        batch_size=16,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print("=== REENTRENAMIENTO COMPLETADO ===")

    return {
        "filas_diferentes": len(df_new),
        "secuencias_entrenadas": len(X_seq),
        "loss_final": float(history.history["loss"][-1]),
        "mae_final": float(history.history["mean_absolute_error"][-1])
    }