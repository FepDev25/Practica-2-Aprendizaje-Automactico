import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 2: Modelado con GRU y MLflow

    En esta fase, usaremos el dataset procesado (`dataset_processed_advanced.csv`)
    para entrenar un modelo de Red Neuronal Recurrente (GRU).

    El objetivo es predecir `quantity_available`.

    Usaremos **MLflow** para registrar y gestionar nuestros experimentos.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    import mlflow
    import mlflow.tensorflow 

    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


    import matplotlib.pyplot as plt
    import seaborn as sns
    import math 
    import os

    # Configuración
    sns.set_theme(style="whitegrid")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    print("Todas las librerías para modelado GRU y MLflow han sido importadas.")
    return (
        Adam,
        Dense,
        Dropout,
        EarlyStopping,
        GRU,
        MinMaxScaler,
        ModelCheckpoint,
        Sequential,
        load_model,
        mean_absolute_error,
        mean_squared_error,
        mlflow,
        np,
        os,
        pd,
        plt,
        r2_score,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Carga de Datos Procesados

    Cargamos el dataset `dataset_processed_advanced.csv`.
    Este dataset ya tiene todos nuestros features (lags, medias, etc.).
    """)
    return


@app.cell
def _(pd):
    # Cargar el dataset final de la Fase 2
    try:
        df_model = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_processed_advanced.csv")

        print(f"Dataset cargado exitosamente. Forma: {df_model.shape}")


        cols_a_excluir = ['product_sku']
        if 'region_almacen' in df_model.columns:
            cols_a_excluir.append('region_almacen')

        df_model = df_model.drop(columns=cols_a_excluir, errors='ignore')

        print(f"Columnas no numéricas ('{cols_a_excluir}') eliminadas.")
        print("\n--- df_model.info() (solo numéricas) ---")
        df_model.info()

    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dataset_processed_advanced.csv'.")
        df_model = pd.DataFrame() 
    return (df_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Escalado de Datos (Normalización)

    Las Redes Neuronales (GRU) **requieren** que todos los datos estén
    normalizados (usualmente entre 0 y 1).

    * `scaler_y`: Escala nuestro target (`quantity_available`).
    * `scaler_X`: Escala todos nuestros 38 features.

    **Importante:** Guardamos los "scalers" para poder *revertir*
    las predicciones al final y entenderlas en unidades reales.
    """)
    return


@app.cell
def _(MinMaxScaler, df_model):
    if not df_model.empty:
        # 1. Definir el Target (y) y los Features (X)
        y = df_model[['quantity_available']] # Mantener como DataFrame
        X = df_model.drop(columns=['quantity_available'])

        N_FEATURES = X.shape[1] # Guardamos el número de features

        # 2. Crear e instanciar los scalers
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_X = MinMaxScaler(feature_range=(0, 1))

        # 3. Ajustar y transformar los datos
        y_scaled = scaler_y.fit_transform(y)
        X_scaled = scaler_X.fit_transform(X)

        print(f"Datos escalados a [0, 1].")
        print(f"Forma de X_scaled: {X_scaled.shape}")
        print(f"Forma de y_scaled: {y_scaled.shape}")
        print(f"Número de features: {N_FEATURES}")
    else:
        print("Dataset vacío, no se puede escalar.")
        X_scaled, y_scaled, scaler_X, scaler_y, N_FEATURES = [None] * 5
    return N_FEATURES, X_scaled, scaler_y, y, y_scaled


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. División Cronológica (Train/Validation/Test)

    Para series de tiempo, debemos
    dividir los datos cronológicamente:
    - **Train (80%):** El pasado más antiguo para aprendizaje
    - **Validation (10%):** Datos recientes para ajuste de hiperparámetros
    - **Test (10%):** El futuro (examen final, datos nunca vistos)

    **¿Por qué esta división?**
    - 80% Train: Suficientes datos para que el modelo aprenda patrones complejos
    - 10% Validation: Permite evaluar generalización sin comprometer datos de prueba
    - 10% Test: Conjunto limpio para evaluación final imparcial

    **Importante:** En series temporales, la división debe ser cronológica (no aleatoria)
    para respetar la secuencia temporal de los datos.
    """)
    return


@app.cell
def _(X_scaled, y_scaled):
    if X_scaled is not None:
        # Calcular los puntos de corte (80/10/10)
        total_size = len(X_scaled)
        train_size = int(total_size * 0.80)  # 80%
        val_size = int(total_size * 0.10)    # 10%
        # El resto es el test_size (10%)

        # Dividir X cronológicamente
        X_train_raw = X_scaled[0:train_size]
        X_val_raw = X_scaled[train_size : train_size + val_size]
        X_test_raw = X_scaled[train_size + val_size :]

        # Dividir y cronológicamente
        y_train_raw = y_scaled[0:train_size]
        y_val_raw = y_scaled[train_size : train_size + val_size]
        y_test_raw = y_scaled[train_size + val_size :]

        print(f" División cronológica completada:")
        print(f"   Total de datos: {total_size:,}")
        print(f"   Datos de Train: {len(X_train_raw):,} ({len(X_train_raw)/total_size*100:.1f}%)")
        print(f"   Datos de Validation: {len(X_val_raw):,} ({len(X_val_raw)/total_size*100:.1f}%)")
        print(f"   Datos de Test: {len(X_test_raw):,} ({len(X_test_raw)/total_size*100:.1f}%)")
        print(f"\n    Nota: La división respeta el orden cronológico de los datos.")
    else:
        print(" Datos no escalados, no se puede dividir.")
        X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = [None] * 6
    return (
        X_test_raw,
        X_train_raw,
        X_val_raw,
        y_test_raw,
        y_train_raw,
        y_val_raw,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Creación de Secuencias

    Aquí convertimos nuestras "filas" en "películas" (secuencias).
    * `N_STEPS = 7`
    * Definimos que 1 "película" = 7 días de historia.
    * El modelo mirará los 7 días de `X` para predecir el `y` del día 8.
    * El formato de entrada final será: `(muestras, 7, 38)`
    """)
    return


@app.cell
def _(
    N_FEATURES,
    X_test_raw,
    X_train_raw,
    X_val_raw,
    np,
    y_test_raw,
    y_train_raw,
    y_val_raw,
):

    N_STEPS = 7 # 7 días de historia para predecir el 8vo

    def create_sequences(X_data, y_data, time_steps=1):
        """Crea secuencias (X, y) para un modelo RNN/GRU."""
        Xs, ys = [], []
        for i in range(len(X_data) - time_steps):
            # X = tomar los N_STEPS (7) siguientes features
            Xs.append(X_data[i : (i + time_steps)])
            # y = tomar el target del día siguiente al final de la secuencia
            ys.append(y_data[i + time_steps])
        return np.array(Xs), np.array(ys)

    if X_train_raw is not None:
        # Crear las secuencias
        X_train, y_train = create_sequences(X_train_raw, y_train_raw, N_STEPS)
        X_val, y_val = create_sequences(X_val_raw, y_val_raw, N_STEPS)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, N_STEPS)

        # Definir el INPUT_SHAPE para la Red Neuronal
        INPUT_SHAPE = (N_STEPS, N_FEATURES)

        print(f"Forma de X_train (secuencias): {X_train.shape}")
        print(f"Forma de y_train (secuencias): {y_train.shape}")
        print(f"Forma de X_val (secuencias): {X_val.shape}")
        print(f"Forma de X_test (secuencias): {X_test.shape}")
        print(f"INPUT_SHAPE para el modelo: {INPUT_SHAPE}")
    else:
        print("No hay datos para crear secuencias.")
        X_train, y_train, X_val, y_val, X_test, y_test, INPUT_SHAPE = [None] * 7
    return INPUT_SHAPE, X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Construcción del Modelo GRU

    Definimos la arquitectura del modelo, siguiendo el ejemplo de la
    práctica anterior.
    * **Capa GRU:** La capa de "memoria" que mira la secuencia de 7 días.
    * **Capa Dropout:** Apaga neuronas al azar para evitar sobreajuste.
    * **Capa Dense:** La capa final que da la predicción (1 número).
    """)
    return


@app.cell
def _(Dense, Dropout, GRU, INPUT_SHAPE, Sequential):
    if INPUT_SHAPE:
        model = Sequential(name="Modelo_GRU_Prediccion_Stock")

        # 1. Capa GRU de entrada
        model.add(GRU(units=64, 
                      input_shape=INPUT_SHAPE, 
                      name="Capa_Entrada_GRU"))

        # 2. Capa Dropout (Regularización)
        model.add(Dropout(0.2, name="Capa_Dropout"))

        # 3. Capa Densa de Salida
        model.add(Dense(units=1, name="Capa_Salida_Prediccion"))

        model.summary()
    else:
        print("No se puede construir el modelo, INPUT_SHAPE no definido.")
        model = None
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Compilación, Callbacks y MLflow

    1.  **Compilamos** el modelo (definimos el optimizador y la pérdida).
    2.  **Callbacks:** Creamos `EarlyStopping` (para que pare si no mejora)
        y `ModelCheckpoint` (para guardar solo la mejor versión).
    3.  **MLflow:** Activamos `autolog()` para registrar todo
        automáticamente.
    """)
    return


@app.cell
def _(Adam, EarlyStopping, ModelCheckpoint, mlflow, model):
    if model:
        # --- 1. Compilar el modelo ---
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error', 
            metrics=['mean_absolute_error']
        )
        print("Modelo compilado.")

        # --- 2. Callbacks ---
        checkpoint_path = 'best_model.keras'
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss', # Monitorea la pérdida en Validación
            save_best_only=True,
            mode='min',
            verbose=1 
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10, # Parar si no mejora en 10 épocas
            mode='min',
            verbose=1,
        )
        print("Callbacks definidos.")

        # --- 3. Configuración de MLflow ---
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Prediccion de Stock GRU")

        # ¡AUTOLOG! Esto registra params, métricas por época, y el modelo
        mlflow.tensorflow.autolog() 
        print("MLflow configurado con autolog para TensorFlow.")

    else:
        print("No hay modelo para compilar.")
        model_checkpoint, early_stopping = None, None
    return early_stopping, model_checkpoint


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Entrenamiento del Modelo

    Iniciamos el entrenamiento. MLflow registrará todo automáticamente.
    """)
    return


@app.cell
def _(X_train, X_val, early_stopping, model, model_checkpoint, y_train, y_val):
    if model:
        EPOCHS = 100      
        # Evitar batch_size mayor que número de muestras
        n_train_samples = X_train.shape[0] if X_train is not None else 0
        BATCH_SIZE = min(64, max(1, n_train_samples))

        callbacks = [cb for cb in (model_checkpoint, early_stopping) if cb is not None]


        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val), 
            callbacks=[model_checkpoint, early_stopping],
            verbose=1 
        )
        print("Entrenamiento completado.")
    else:
        print("No hay modelo para entrenar.")
        history = None
    return (history,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Visualización de Curvas de Aprendizaje

    Veamos cómo el modelo aprendió (Loss) y mejoró (MAE) en cada época,
    tanto en los datos de entrenamiento (azul) como en los de validación (naranja).
    """)
    return


@app.cell
def _(history, mlflow, plt):
    if history:
        loss = history.history.get('loss') or history.history.get('Loss')
        val_loss = history.history.get('val_loss') or history.history.get('val_Loss')

        # fallback para nombres de MAE
        mae = history.history.get('mean_absolute_error') or history.history.get('mae')
        val_mae = history.history.get('val_mean_absolute_error') or history.history.get('val_mae')

        epochs_range = range(len(loss))

        plt.figure(figsize=(14, 6))

        # Gráfico de Pérdida (Loss - MSE)
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento (MSE)')
        plt.plot(epochs_range, val_loss, label='Pérdida de Validación (MSE)')
        plt.legend(loc='upper right')
        plt.title('Pérdida (Loss) de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (MSE)')

        # Gráfico de Métrica (MAE)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, mae, label='Error Absoluto Medio (MAE) de Entrenamiento')
        plt.plot(epochs_range, val_mae, label='Error Absoluto Medio (MAE) de Validación')
        plt.legend(loc='upper right')
        plt.title('Métrica (MAE) de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Error (MAE)')

        # Guardar y loguear el gráfico
        learning_plot_path = "learning_curves.png"
        plt.savefig(learning_plot_path)
        plt.show()

        try:
            mlflow.log_artifact(learning_plot_path)
            print(f"Curvas de aprendizaje guardadas y logueadas en MLflow.")
        except Exception as e:
            print(f"No se pudo loguear curvas de aprendizaje en MLflow: {e}")

    else:
        print("No hay historial para graficar.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Análisis de las curvas de aprendizaje:**
    1. Pérdida (Loss) vs. Épocas:

        - La línea azul (Entrenamiento) baja rápidamente y se estabiliza. La línea naranja (Validación) es aún mejor: es extremadamente baja (casi cero) y plana.

    **Conclusión**: Cero Overfitting

    El hecho de que la pérdida de validación (naranja) sea tan baja significa que nuestro modelo generaliza perfectamente a datos que nunca ha visto. No hay sobreajuste en absoluto. El EarlyStopping funcionó bien y detuvo el entrenamiento en el momento justo (alrededor de la época 40).


    Es un modelo saludable, bien entrenado y que generaliza correctamente.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Evaluación Final del Modelo (en Unidades Reales)

    Cargamos el **mejor modelo** (guardado por `ModelCheckpoint`) y lo
    probamos en el "examen final" (`X_test`).

    **Lo más importante:** Revertimos la escala (de [0,1] a unidades reales)
    para entender el error (MAE) en *unidades de stock*.
    """)
    return


@app.cell
def _(
    X_test,
    load_model,
    mean_absolute_error,
    mean_squared_error,
    mlflow,
    mo,
    np,
    os,
    pd,
    plt,
    r2_score,
    scaler_y,
    sns,
    y_test,
):
    # Fase 9: Evaluación Final (Variables renombradas para evitar conflicto)
    # 1. Validaciones
    mo.stop(
        X_test is None or len(X_test) == 0,
        mo.md(" **Detenido:** No hay datos de prueba (`X_test` es None o está vacío).")
    )

    model_path = 'best_model.keras'
    mo.stop(
        not os.path.exists(model_path),
        mo.md(f" **Error:** No se encontró el archivo '{model_path}'.")
    )

    # 2. Predecir
    best_model = load_model(model_path)
    y_pred_scaled = best_model.predict(X_test)

    # 3. Invertir escala
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_real = scaler_y.inverse_transform(y_test)

    # 4. Calcular métricas (RENOMBRADAS con el prefijo 'test_')
    # Esto soluciona el conflicto con la Celda 17
    test_mae = mean_absolute_error(y_test_real, y_pred)
    test_mse = mean_squared_error(y_test_real, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_real, y_pred)

    print("\n--- Métricas del 'Examen Final' (en Unidades Reales) ---")
    print(f" R² (R-squared):    {test_r2:.3f}")
    print(f" MAE: {test_mae:.3f} unidades")
    print(f" RMSE: {test_rmse:.3f} unidades")

    # 5. DataFrame de Resultados
    results_df = pd.DataFrame({
        'Real': y_test_real.flatten(), 
        'Predicción': y_pred.flatten()
    })
    results_df['Error'] = results_df['Real'] - results_df['Predicción']

    # 6. Visualización
    fig_eval = plt.figure(figsize=(20, 8))

    # Scatter Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Real', y='Predicción', data=results_df, alpha=0.6, s=50)
    max_val = max(results_df['Real'].max(), results_df['Predicción'].max())
    min_val = min(results_df['Real'].min(), results_df['Predicción'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    plt.title('Valores Reales vs. Predicciones', fontsize=16)
    plt.xlabel('Stock Disponible Real')
    plt.ylabel('Stock Predicho')
    plt.legend()
    plt.grid(True)

    # Histograma
    plt.subplot(1, 2, 2)
    sns.histplot(results_df['Error'], kde=True, bins=30)
    plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Error Cero')
    plt.title('Distribución de Errores (Residuals)', fontsize=16)
    plt.xlabel('Error (Real - Predicción)')
    plt.legend()

    plot_path = "evaluation_plots.png"
    plt.savefig(plot_path)
    plt.tight_layout()

    # Gráfico de métricas (Usando las nuevas variables)
    fig_metrics = plt.figure(figsize=(6,4))
    plt.bar(['MAE','RMSE','R2'], [test_mae, test_rmse, test_r2], color=['tab:blue','tab:orange','tab:green'])
    plt.title('Resumen de Métricas (Test)')
    metrics_plot_path = "metrics_summary.png"
    plt.savefig(metrics_plot_path)
    plt.close(fig_metrics) 

    # 7. Loguear a MLflow
    try:
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            mlflow.log_metrics({
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2)
            })
            mlflow.log_artifact(plot_path)
            mlflow.log_artifact(metrics_plot_path)
        print(" Métricas de Test logueadas correctamente.")
    except Exception as e:
        print(f" Advertencia MLflow: {e}")

    fig_eval
    return test_rmse, y_pred_scaled


@app.cell
def _(mo):
    mo.md(r"""
    **Análisis de los Resultados:**

    1. R² (R-squared): 0.966

    Un R² de 0.966 significa que nuestro modelo GRU es capaz de explicar el 96.6% del comportamiento (varianza) del stock disponible. Es decir, el modelo entiende casi perfectamente por qué el stock sube o baja.

    2. MAE (Error Absoluto Medio): 57.341 unidades

    En promedio, cuando el modelo hace una predicción, se equivoca por 57 unidades (ya sea hacia arriba o hacia abajo).

    3. RMSE (Raíz del Error Cuadrático Medio): 75.865 unidades

    Esta es la métrica estándar para el error. Es más sensible a errores grandes que el MAE. Este valor es el que usamos para la "Regla del 10%".

    4. Gráfico (Izquierda - Reales vs. Predicciones):

    Podemos observar que es casi una línea recta perfecta. Los puntos azules (predicciones) están increíblemente agrupados alrededor de la línea roja (la predicción perfecta). Esto demuestra que el modelo es preciso en todo el rango, tanto para valores bajos de stock como para valores altos.

    5. Gráfico (Derecha - Distribución de Errores):
    Esta es la "campana de Gauss" ideal. Está perfectamente centrada en el cero, lo que nos dice que el modelo no tiene sesgo. No tiende a predecir "siempre de más" o "siempre de menos"
    """)
    return


@app.cell
def _(mo, test_rmse, y):
    # Cálculo de la Regla del 10% (Usando test_rmse)

    # 1. Calcular el promedio de nuestro target (en unidades reales)
    # 'y' viene de tu celda de Escalado (Celda 2 o 3)
    y_mean = y['quantity_available'].mean()

    # 2. Calcular el error porcentual
    # Usamos 'test_rmse' que definimos en la celda anterior
    error_porcentual = (test_rmse / y_mean) * 100

    # 3. Mostrar resultado en Markdown dinámico
    mo.md(rf"""
    ###  Calificación del Modelo: La Regla del 10%

    Comparamos nuestro error (RMSE) con el promedio histórico del stock para saber qué tan grave es el error.

    * **Promedio del Stock (y):** `{y_mean:.2f}` unidades
    * **Error del Modelo (RMSE):** `{test_rmse:.2f}` unidades
    * **Error Porcentual:** `({test_rmse:.2f} / {y_mean:.2f}) * 100` = **`{error_porcentual:.2f}%`**

    ---
    **Interpretación:**
    *  **< 10%:** Excelente (Alta precisión).
    *  **10% - 20%:** Bueno (Útil para decisiones de negocio).
    *  **> 20%:** Regular (El modelo puede confundirse con la volatilidad).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Análisis: en la industria, un error por debajo del 10-15% se considera muy bueno. Un error de menos del 5% es excepcional.

    En otras palabras, el modelo tiene una precisión superior al 95% (100% - 4.94% = 95.06%) en relación al comportamiento promedio del stock.
    """)
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, mo, np, y_pred_scaled, y_test):
    # Cálculo de errores en escala normalizada (0 a 1)
    # ------------------------------------------------

    # 1. Calculamos las métricas usando los datos SIN inverse_transform
    # y_test: son los valores reales escalados (vienen de create_sequences)
    # y_pred_scaled: son las predicciones crudas del modelo (0 a 1)

    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
    mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    rmse_scaled = np.sqrt(mse_scaled)

    mo.md(rf"""
    ### Errores en Escala Normalizada (0 - 1)

    Estos son los valores de pérdida que la Red Neuronal intentó minimizar durante el entrenamiento.

    * **MAE Escalado:** `{mae_scaled:.4f}`
    * **MSE Escalado:** `{mse_scaled:.4f}`
    * **RMSE Escalado:** `{rmse_scaled:.4f}`

    **Interpretación:**
    Un MAE escalado de `{mae_scaled:.4f}` significa que, en el espacio matemático de 0 a 1, el modelo se desvía en promedio un **{mae_scaled*100:.2f}%** del rango total de tus datos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Análisis de los Resultados:**
    un error de 0.0095 en esa escala significa que el error promedio del modelo es inferior al 1% del rango total de los datos. Es otra forma de confirmar, a nivel matemático, que la precisión es altísima. El modelo apenas comete errores significativos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Comenzamos ML Flow
    Para ver el dashboard de nuestros resultados:

    1.  Abre una **nueva terminal**
    2.  Navega (`cd`) a la carpeta de tu proyecto (donde está este cuaderno
        y la nueva carpeta `mlruns`).
    3.  Ejecuta el siguiente comando:

    ```bash
    mlflow ui
    ```

    4.  Abre el navegador web y ve a la dirección que te indica
        (usualmente `http://127.0.0.1:5000`).
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
