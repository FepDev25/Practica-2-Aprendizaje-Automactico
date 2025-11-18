# filepath: c:\Users\samil\Desktop\APRENDIZAJE AUTOMATICO\PRIMER INTERCICLO\Practica-2-Aprendizaje-Automactico\modelo\notebooks\Fase_03.py

import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 3: Optimización de Hiperparámetros con Grid Search

    En esta fase, realizaremos **fine-tuning** del modelo GRU utilizando **Grid Search**.
    En la **Fase 2** entrenamos un modelo GRU con hiperparámetros "elegidos a ojo".
    Ahora vamos a ser más científicos: probaremos **108 combinaciones diferentes**
    de configuraciones para encontrar la **mejor**.

    ## Objetivos:
    1. Definir un espacio de búsqueda de hiperparámetros
    2. Implementar Grid Search manualmente para modelos Keras
    3. Entrenar múltiples modelos y comparar resultados
    4. Registrar todos los experimentos en MLflow
    5. Seleccionar el mejor modelo basado en métricas de validación
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
    from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    import json
    import os
    from datetime import datetime

    # Configuración
    sns.set_theme(style="whitegrid")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    print(" Todas las librerías para Grid Search y MLflow han sido importadas.")
    return (
        Adam,
        Dense,
        Dropout,
        EarlyStopping,
        GRU,
        LSTM,
        MinMaxScaler,
        ModelCheckpoint,
        Sequential,
        datetime,
        itertools,
        json,
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
    - Utilizamos el mismo dataset de la Fase 2.
    - Las columnas no numéricas causan errores en modelos de Machine Learning
    - Validamos que los datos estén disponibles antes de continuar
    """)
    return


@app.cell
def _(pd):
    try:
        df_model = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_processed_advanced.csv")
        print(f"Dataset cargado exitosamente. Forma: {df_model.shape}")

        cols_a_excluir = ['product_sku']
        if 'region_almacen' in df_model.columns:
            cols_a_excluir.append('region_almacen')

        df_model = df_model.drop(columns=cols_a_excluir, errors='ignore')
        print(f"✓ Columnas no numéricas eliminadas: {cols_a_excluir}")

    except FileNotFoundError:
        print(" Error: No se encontró el archivo 'dataset_processed_advanced.csv'.")
        df_model = pd.DataFrame() 
    return (df_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Escalado de Datos
    Normalizamos los datos entre 0 y 1.
    """)
    return


@app.cell
def _(MinMaxScaler, df_model):
    if not df_model.empty:
        y = df_model[['quantity_available']]
        X = df_model.drop(columns=['quantity_available'])
        N_FEATURES = X.shape[1]

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_X = MinMaxScaler(feature_range=(0, 1))

        y_scaled = scaler_y.fit_transform(y)
        X_scaled = scaler_X.fit_transform(X)

        print(f"✓ Datos escalados correctamente.")
        print(f"  - Forma de X_scaled: {X_scaled.shape}")
        print(f"  - Número de features: {N_FEATURES}")
    else:
        X_scaled, y_scaled, scaler_X, scaler_y, N_FEATURES = [None] * 5
    return N_FEATURES, X_scaled, scaler_y, y, y_scaled


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. División Cronológica (Train/Validation/Test)

    **Mantenemos la misma división de la Fase 2 para comparabilidad:**
    - **Train (80%):** Datos históricos para entrenamiento del modelo
    - **Validation (10%):** Conjunto para selección de hiperparámetros en Grid Search
    - **Test (10%):** Evaluación final imparcial (nunca visto durante optimización)

    **¿Por qué es crucial mantener la misma división?**
    1. **Comparabilidad:** Permite comparar directamente Baseline vs Grid Search
    2. **Validez:** El test set debe ser idéntico en ambos experimentos
    3. **Integridad:** Evita "data leakage" (filtración de información)

    **Flujo de datos:**
    ```
    [──────── Train (80%) ────────][Val(10%)][Test(10%)]
                                      ↓           ↓
                              Grid Search    Evaluación
                              Optimización      Final
    ```
    """)
    return


@app.cell
def _(X_scaled, y_scaled):
    if X_scaled is not None:
        total_size = len(X_scaled)
        train_size = int(total_size * 0.80)  # 80% para train
        val_size = int(total_size * 0.10)    # 10% para validation

        # División cronológica (NO aleatoria)
        X_train_raw = X_scaled[0:train_size]
        X_val_raw = X_scaled[train_size : train_size + val_size]
        X_test_raw = X_scaled[train_size + val_size :]

        y_train_raw = y_scaled[0:train_size]
        y_val_raw = y_scaled[train_size : train_size + val_size]
        y_test_raw = y_scaled[train_size + val_size :]

        print(f" División 80/10/10 completada:")
        print(f"    Total: {total_size:,} muestras")
        print(f"    Train: {len(X_train_raw):,} ({len(X_train_raw)/total_size*100:.1f}%)")
        print(f"    Validation: {len(X_val_raw):,} ({len(X_val_raw)/total_size*100:.1f}%)")
        print(f"    Test: {len(X_test_raw):,} ({len(X_test_raw)/total_size*100:.1f}%)")
        print(f"\n    División idéntica a Fase 2 para comparación válida.")
    else:
        print(" Error: Datos no escalados.")
        X_train_raw, X_val_raw, X_test_raw = [None] * 3
        y_train_raw, y_val_raw, y_test_raw = [None] * 3
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
    Convertimos los datos en secuencias temporales para el modelo GRU.
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
    N_STEPS = 7

    def create_sequences(X_data, y_data, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X_data) - time_steps):
            Xs.append(X_data[i : (i + time_steps)])
            ys.append(y_data[i + time_steps])
        return np.array(Xs), np.array(ys)

    if X_train_raw is not None:
        X_train, y_train = create_sequences(X_train_raw, y_train_raw, N_STEPS)
        X_val, y_val = create_sequences(X_val_raw, y_val_raw, N_STEPS)
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, N_STEPS)

        INPUT_SHAPE = (N_STEPS, N_FEATURES)

        print(f"✓ Secuencias creadas:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - X_val: {X_val.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - INPUT_SHAPE: {INPUT_SHAPE}")
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, INPUT_SHAPE = [None] * 7
    return INPUT_SHAPE, X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Definición del Espacio de Hiperparámetros

    Definimos los hiperparámetros a explorar en nuestro Grid Search:

    | Hiperparámetro | Valores a Probar | Descripción |
    |----------------|------------------|-------------|
    | **units** | [32, 64, 128] | Neuronas en la capa GRU |
    | **dropout** | [0.1, 0.2, 0.3] | Tasa de regularización |
    | **learning_rate** | [0.001, 0.0005, 0.0001] | Velocidad de aprendizaje |
    | **batch_size** | [32, 64] | Tamaño del lote |
    | **layer_type** | ['GRU', 'LSTM'] | Tipo de capa recurrente |

    **Total de combinaciones:** 3 × 3 × 3 × 2 × 2 = **108 modelos**
    """)
    return


@app.cell
def _(itertools, json):
    # Definición del espacio de búsqueda
    param_grid = {
        'units': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64],
        'layer_type': ['GRU', 'LSTM']
    }

    # Generar todas las combinaciones posibles
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"✓ Espacio de búsqueda definido:")
    print(f"  - Total de combinaciones: {len(param_combinations)}")
    print(f"\n Ejemplo de combinación:")
    print(json.dumps(param_combinations[0], indent=2))
    return (param_combinations,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Función para Construir Modelos

    Creamos una función que construye un modelo GRU/LSTM con los hiperparámetros especificados.
    """)
    return


@app.cell
def _(Adam, Dense, Dropout, GRU, INPUT_SHAPE, LSTM, Sequential):
    def build_model(units, dropout, learning_rate, layer_type='GRU'):
        """
        Construye un modelo GRU o LSTM con los hiperparámetros dados.

        Args:
            units (int): Número de neuronas en la capa recurrente
            dropout (float): Tasa de dropout (0.0 - 1.0)
            learning_rate (float): Tasa de aprendizaje del optimizador
            layer_type (str): 'GRU' o 'LSTM'

        Returns:
            model: Modelo Keras compilado
        """
        model = Sequential(name=f"Modelo_{layer_type}_Optimizado")

        # Capa recurrente (GRU)
        if layer_type == 'GRU':
            model.add(GRU(units=units, 
                          input_shape=INPUT_SHAPE, 
                          name=f"Capa_{layer_type}"))
        else:  # LSTM
            model.add(LSTM(units=units, 
                           input_shape=INPUT_SHAPE, 
                           name=f"Capa_{layer_type}"))

        # Capa Dropout
        model.add(Dropout(dropout, name="Capa_Dropout"))

        # Capa de salida
        model.add(Dense(units=1, name="Capa_Salida"))

        # Compilar
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model

    print("✓ Función build_model() definida correctamente.")
    return (build_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Grid Search Manual con MLflow

    Implementamos el Grid Search manualmente, entrenando cada combinación
    y registrando los resultados en MLflow.
    """)
    return


@app.cell
def _(
    EarlyStopping,
    ModelCheckpoint,
    X_train,
    X_val,
    build_model,
    datetime,
    mlflow,
    np,
    os,
    param_combinations,
    y_train,
    y_val,
):
    # Configuración de MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Grid_Search_GRU_LSTM_Optimizacion")

    # Parámetros de entrenamiento
    EPOCHS_GS = 50  # Reducido para acelerar Grid Search
    PATIENCE = 10

    # Almacenar resultados
    results_grid_search = []

    print(f" Iniciando Grid Search...")
    print(f"  Tiempo estimado: ~{len(param_combinations) * 2} minutos (aprox.)\n")

    # Iterar sobre todas las combinaciones
    for idx, params in enumerate(param_combinations, 1):

        print(f"\n{'='*70}")
        print(f" Experimento {idx}/{len(param_combinations)}")
        print(f" Parámetros: {params}")
        print(f"{'='*70}")

        # Iniciar run de MLflow
        with mlflow.start_run(run_name=f"GridSearch_{idx}_{params['layer_type']}"):

            # Loguear hiperparámetros
            mlflow.log_params(params)

            # Construir modelo
            model_gs = build_model(
                units=params['units'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate'],
                layer_type=params['layer_type']
            )

            # Callbacks
            checkpoint_path_gs = f'models/grid_search_model_{idx}.keras'
            os.makedirs('models', exist_ok=True)

            callbacks_gs = [
                EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', verbose=0),
                ModelCheckpoint(filepath=checkpoint_path_gs, monitor='val_loss', 
                                save_best_only=True, mode='min', verbose=0)
            ]

            # Entrenar
            start_time = datetime.now()

            history_gs = model_gs.fit(
                X_train, y_train,
                epochs=EPOCHS_GS,
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks_gs,
                verbose=0  # Sin output para acelerar
            )

            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()

            # Obtener mejor época
            best_epoch = np.argmin(history_gs.history['val_loss']) + 1
            best_val_loss = np.min(history_gs.history['val_loss'])
            best_val_mae = history_gs.history['val_mean_absolute_error'][best_epoch - 1]

            # Loguear métricas
            mlflow.log_metrics({
                'best_val_loss': float(best_val_loss),
                'best_val_mae': float(best_val_mae),
                'best_epoch': int(best_epoch),
                'training_duration_seconds': float(training_duration)
            })

            # Loguear modelo
            mlflow.keras.log_model(model_gs, "model")

            # Guardar resultados
            result = {
                'experiment_id': idx,
                'params': params,
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'best_epoch': best_epoch,
                'training_duration': training_duration
            }
            results_grid_search.append(result)

            print(f"Mejor val_loss: {best_val_loss:.6f} (época {best_epoch})")
            print(f"Mejor val_MAE: {best_val_mae:.6f}")
            print(f"  Duración: {training_duration:.1f}s")

    print(f"\n{'='*70}")
    print(f" Grid Search completado exitosamente!")
    print(f" {len(results_grid_search)} modelos entrenados y registrados en MLflow")
    print(f"{'='*70}\n")
    return (results_grid_search,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Análisis de Resultados del Grid Search

    Analizamos y visualizamos los resultados de todas las combinaciones probadas.
    """)
    return


@app.cell
def _():
    import subprocess
    import sys

    # Instalar tabulate en el entorno actual
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])

    print(" tabulate instalado correctamente")
    print(" Ahora ejecuta la celda problemática nuevamente")
    return


@app.cell
def _(mo, pd, results_grid_search):
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame([
        {
            'experiment_id': r['experiment_id'],
            'units': r['params']['units'],
            'dropout': r['params']['dropout'],
            'learning_rate': r['params']['learning_rate'],
            'batch_size': r['params']['batch_size'],
            'layer_type': r['params']['layer_type'],
            'best_val_loss': r['best_val_loss'],
            'best_val_mae': r['best_val_mae'],
            'best_epoch': r['best_epoch'],
            'training_duration': r['training_duration']
        }
        for r in results_grid_search
    ])

    # Ordenar por mejor val_loss
    results_df_sorted = results_df.sort_values('best_val_loss').reset_index(drop=True)

    mo.md(f"""
    ###  Tabla de Resultados (Top 10 mejores modelos)

    {results_df_sorted.head(10).to_markdown(index=False)}

    **Total de modelos evaluados:** {len(results_df_sorted)}
    """)
    return (results_df_sorted,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Visualización de Resultados
    """)
    return


@app.cell
def _(mlflow, plt, results_df_sorted, sns):
    # Crear visualizaciones
    fig_grid = plt.figure(figsize=(20, 12))

    # 1. Box plot: Val Loss por tipo de capa
    plt.subplot(2, 3, 1)
    sns.boxplot(data=results_df_sorted, x='layer_type', y='best_val_loss', palette='Set2')
    plt.title('Distribución de Val Loss por Tipo de Capa', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss (MSE)')
    plt.xlabel('Tipo de Capa')

    # 2. Box plot: Val Loss por número de unidades
    plt.subplot(2, 3, 2)
    sns.boxplot(data=results_df_sorted, x='units', y='best_val_loss', palette='Set1')
    plt.title('Val Loss por Número de Unidades', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss (MSE)')
    plt.xlabel('Unidades')

    # 3. Box plot: Val Loss por dropout
    plt.subplot(2, 3, 3)
    sns.boxplot(data=results_df_sorted, x='dropout', y='best_val_loss', palette='Set3')
    plt.title('Val Loss por Tasa de Dropout', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Loss (MSE)')
    plt.xlabel('Dropout')

    # 4. Scatter: Learning rate vs Val Loss
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=results_df_sorted, x='learning_rate', y='best_val_loss', 
                    hue='layer_type', size='units', sizes=(50, 300), alpha=0.6)
    plt.title('Learning Rate vs Val Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss (MSE)')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. Bar plot: Top 10 mejores modelos
    plt.subplot(2, 3, 5)
    top_10 = results_df_sorted.head(10)
    colors_top = ['green' if i == 0 else 'skyblue' for i in range(len(top_10))]
    plt.barh(range(len(top_10)), top_10['best_val_loss'], color=colors_top)
    plt.yticks(range(len(top_10)), 
               [f"Exp {row['experiment_id']}\n{row['layer_type']}" for _, row in top_10.iterrows()])
    plt.xlabel('Validation Loss (MSE)')
    plt.title('Top 10 Mejores Modelos', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # 6. Heatmap: Promedio de Val Loss por units y dropout
    plt.subplot(2, 3, 6)
    pivot_table = results_df_sorted.pivot_table(
        values='best_val_loss', 
        index='dropout', 
        columns='units', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Avg Val Loss'})
    plt.title('Mapa de Calor: Units vs Dropout', fontsize=14, fontweight='bold')
    plt.ylabel('Dropout')
    plt.xlabel('Units')

    plt.tight_layout()

    # Guardar y loguear
    grid_plot_path = "grid_search_analysis.png"
    plt.savefig(grid_plot_path, dpi=150, bbox_inches='tight')

    try:
        with mlflow.start_run(run_name="Grid_Search_Summary"):
            mlflow.log_artifact(grid_plot_path)
            mlflow.log_metrics({
                'best_overall_val_loss': float(results_df_sorted.iloc[0]['best_val_loss']),
                'best_overall_val_mae': float(results_df_sorted.iloc[0]['best_val_mae'])
            })
        print("✓ Gráficos de análisis logueados en MLflow")
    except Exception as e:
        print(f"  Advertencia al loguear en MLflow: {e}")

    fig_grid
    return


@app.cell
def _(mo, results_df_sorted):
    # Obtener el mejor modelo
    best_model_params = results_df_sorted.iloc[0]

    mo.md(rf"""
    ##  Mejor Modelo Encontrado

    ### Hiperparámetros Óptimos:

    | Parámetro | Valor |
    |-----------|-------|
    | **Tipo de Capa** | {best_model_params['layer_type']} |
    | **Unidades** | {best_model_params['units']} |
    | **Dropout** | {best_model_params['dropout']:.2f} |
    | **Learning Rate** | {best_model_params['learning_rate']:.4f} |
    | **Batch Size** | {best_model_params['batch_size']} |

    ### Métricas de Validación:

    * **Val Loss (MSE):** {best_model_params['best_val_loss']:.6f}
    * **Val MAE:** {best_model_params['best_val_mae']:.6f}
    * **Mejor Época:** {best_model_params['best_epoch']}
    * **Duración de Entrenamiento:** {best_model_params['training_duration']:.1f} segundos

    ---

    ###  Interpretación:

    Este modelo representa la **configuración óptima** encontrada entre las {len(results_df_sorted)} 
    combinaciones evaluadas. Ha demostrado el mejor balance entre:
    - Capacidad de generalización (val_loss mínimo)
    - Precisión en las predicciones (val_MAE bajo)
    - Eficiencia computacional
    """)
    return (best_model_params,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Evaluación del Mejor Modelo en Test Set

    Cargamos el mejor modelo y lo evaluamos en el conjunto de prueba (el "examen final").
    """)
    return


@app.cell
def _(
    X_test,
    best_model_params,
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
    # Validaciones
    mo.stop(
        X_test is None or len(X_test) == 0,
        mo.md(" **Detenido:** No hay datos de prueba (`X_test` es None o está vacío).")
    )

    # Cargar el mejor modelo
    best_model_path = f'models/grid_search_model_{int(best_model_params["experiment_id"])}.keras'

    mo.stop(
        not os.path.exists(best_model_path),
        mo.md(f" **Error:** No se encontró el archivo '{best_model_path}'.")
    )

    # Cargar y predecir
    best_model_final = load_model(best_model_path)
    y_pred_best_scaled = best_model_final.predict(X_test)

    # Invertir escala
    y_pred_best = scaler_y.inverse_transform(y_pred_best_scaled)
    y_test_best_real = scaler_y.inverse_transform(y_test)

    # Calcular métricas finales
    final_mae = mean_absolute_error(y_test_best_real, y_pred_best)
    final_mse = mean_squared_error(y_test_best_real, y_pred_best)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(y_test_best_real, y_pred_best)

    print("\n" + "="*70)
    print(" EVALUACIÓN FINAL DEL MEJOR MODELO EN TEST SET")
    print("="*70)
    print(f" R² (R-squared):     {final_r2:.4f}")
    print(f" MAE:                {final_mae:.3f} unidades")
    print(f" RMSE:               {final_rmse:.3f} unidades")
    print("="*70 + "\n")

    # DataFrame de resultados
    results_best_df = pd.DataFrame({
        'Real': y_test_best_real.flatten(),
        'Predicción': y_pred_best.flatten()
    })
    results_best_df['Error'] = results_best_df['Real'] - results_best_df['Predicción']
    results_best_df['Error_Abs'] = results_best_df['Error'].abs()
    results_best_df['Error_Porcentual'] = (results_best_df['Error_Abs'] / results_best_df['Real']) * 100

    # Visualización
    fig_best = plt.figure(figsize=(20, 10))

    # 1. Scatter Plot
    plt.subplot(2, 3, 1)
    sns.scatterplot(x='Real', y='Predicción', data=results_best_df, alpha=0.6, s=60, color='mediumseagreen')
    max_val_best = max(results_best_df['Real'].max(), results_best_df['Predicción'].max())
    min_val_best = min(results_best_df['Real'].min(), results_best_df['Predicción'].min())
    plt.plot([min_val_best, max_val_best], [min_val_best, max_val_best], 'r--', lw=2.5, label='Predicción Perfecta')
    plt.title('Valores Reales vs Predicciones (Mejor Modelo)', fontsize=14, fontweight='bold')
    plt.xlabel('Stock Real')
    plt.ylabel('Stock Predicho')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Histograma de Errores
    plt.subplot(2, 3, 2)
    sns.histplot(results_best_df['Error'], kde=True, bins=40, color='coral')
    plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Error Cero')
    plt.title('Distribución de Errores', fontsize=14, fontweight='bold')
    plt.xlabel('Error (Real - Predicción)')
    plt.ylabel('Frecuencia')
    plt.legend()

    # 3. Residual Plot
    plt.subplot(2, 3, 3)
    sns.scatterplot(x=results_best_df['Predicción'], y=results_best_df['Error'], alpha=0.5, color='steelblue')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.title('Gráfico de Residuales', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Residual (Error)')
    plt.grid(True, alpha=0.3)

    # 4. Box Plot de Errores
    plt.subplot(2, 3, 4)
    sns.boxplot(y=results_best_df['Error'], color='lightblue')
    plt.title('Distribución de Errores (BoxPlot)', fontsize=14, fontweight='bold')
    plt.ylabel('Error')

    # 5. Q-Q Plot (Normalidad de Residuales)
    plt.subplot(2, 3, 5)
    from scipy import stats
    stats.probplot(results_best_df['Error'], dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normalidad de Residuales)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 6. Métricas Resumen
    plt.subplot(2, 3, 6)
    metrics_labels = ['R²', 'MAE', 'RMSE']
    metrics_values = [final_r2, final_mae/100, final_rmse/100]  # Normalizado para visualización
    colors_metrics = ['green', 'orange', 'red']
    bars = plt.bar(metrics_labels, metrics_values, color=colors_metrics, alpha=0.7)
    plt.title('Métricas de Evaluación (Test)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor')
    for bar, val in zip(bars, [final_r2, final_mae, final_rmse]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Guardar
    best_eval_path = "best_model_evaluation.png"
    plt.savefig(best_eval_path, dpi=150, bbox_inches='tight')

    # Loguear en MLflow
    try:
        with mlflow.start_run(run_name="Best_Model_Final_Evaluation"):
            mlflow.log_params({
                'model_type': best_model_params['layer_type'],
                'units': int(best_model_params['units']),
                'dropout': float(best_model_params['dropout']),
                'learning_rate': float(best_model_params['learning_rate']),
                'batch_size': int(best_model_params['batch_size'])
            })
            mlflow.log_metrics({
                'test_mae': float(final_mae),
                'test_rmse': float(final_rmse),
                'test_r2': float(final_r2),
                'test_mse': float(final_mse)
            })
            mlflow.log_artifact(best_eval_path)
            mlflow.keras.log_model(best_model_final, "best_model_final")
        print("✓ Evaluación final logueada en MLflow")
    except Exception as e:
        print(f"  Advertencia MLflow: {e}")

    fig_best
    return final_mae, final_r2, final_rmse


@app.cell
def _(final_rmse, mo, y):
    # Regla del 10% con el mejor modelo
    y_mean_best = y['quantity_available'].mean()
    error_porcentual_best = (final_rmse / y_mean_best) * 100

    mo.md(rf"""
    ##  Calificación del Mejor Modelo: Regla del 10%

    | Métrica | Valor |
    |---------|-------|
    | **Promedio del Stock** | {y_mean_best:.2f} unidades |
    | **RMSE del Modelo** | {final_rmse:.2f} unidades |
    | **Error Porcentual** | **{error_porcentual_best:.2f}%** |

    ---

    ###  Interpretación:

    {
        " **EXCELENTE** (< 10%): El modelo tiene precisión de nivel profesional para gestión de inventario automática." 
        if error_porcentual_best < 10 
        else "  **BUENO** (10-20%): El modelo es útil para toma de decisiones, pero requiere supervisión humana." 
        if error_porcentual_best < 20 
        else " **REGULAR** (> 20%): El modelo puede confundirse con la volatilidad del stock."
    }

    **Conclusión:** Con un error del **{error_porcentual_best:.2f}%**, el modelo optimizado mediante Grid Search 
    {'ha mejorado significativamente' if error_porcentual_best < 10 else 'presenta un rendimiento aceptable para'} 
    la predicción de stock disponible.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Comparación: Modelo Baseline vs Mejor Modelo

    Comparamos el rendimiento del modelo original (Fase 2) con el mejor modelo encontrado en el Grid Search.
    """)
    return


@app.cell
def _(final_mae, final_r2, final_rmse, mo, pd, plt):
    mo.md(r"""
    ##  Comparación: Baseline (Fase 2) vs Mejor Modelo (Grid Search)

    **¿El Grid Search realmente mejoró el modelo?**

    Comparamos las métricas del modelo original (donde adivinamos los hiperparámetros)
    con el mejor modelo encontrado por Grid Search (búsqueda exhaustiva).
    """)

    #
    # DATOS REALES DE TU FASE 2

    baseline_r2 = 0.966      # R² de la Fase 2
    baseline_mae = 57.341    # MAE de la Fase 2 (unidades)
    baseline_rmse = 75.865   # RMSE de la Fase 2 (unidades)


    comparison_data = {
        'Modelo': ['Baseline (Fase 2)', 'Mejor Modelo (Grid Search)'],
        'R²': [baseline_r2, final_r2],
        'MAE': [baseline_mae, final_mae],
        'RMSE': [baseline_rmse, final_rmse]
    }

    comparison_df = pd.DataFrame(comparison_data)


    mejora_r2 = ((final_r2 - baseline_r2) / baseline_r2) * 100
    mejora_mae = ((baseline_mae - final_mae) / baseline_mae) * 100
    mejora_rmse = ((baseline_rmse - final_rmse) / baseline_rmse) * 100


    mo.md(rf"""
    ###  Tabla Comparativa

    | Modelo | R² | MAE (unidades) | RMSE (unidades) |
    |--------|-----|----------------|-----------------|
    | **Baseline (Fase 2)** | {baseline_r2:.4f} | {baseline_mae:.2f} | {baseline_rmse:.2f} |
    | **Grid Search (Optimizado)** | {final_r2:.4f} | {final_mae:.2f} | {final_rmse:.2f} |

    ---

    ###  Mejoras Porcentuales

    | Métrica | Mejora |
    |---------|--------|
    | **R²** | {mejora_r2:+.2f}% {'bien' if mejora_r2 > 0 else 'alerta'} |
    | **MAE** | {mejora_mae:+.2f}% {'bien' if mejora_mae > 0 else 'alerta'} |
    | **RMSE** | {mejora_rmse:+.2f}% {'bien' if mejora_rmse > 0 else 'alerta'} |

    ---

    ###  Interpretación de Resultados

    {f'''
    - **MAE mejoró en {mejora_mae:.2f}%:** El error promedio se redujo de {baseline_mae:.2f} a {final_mae:.2f} unidades.
    - **RMSE mejoró en {mejora_rmse:.2f}%:** El error general bajó de {baseline_rmse:.2f} a {final_rmse:.2f} unidades.
    - **R² {'mejoró' if mejora_r2 > 0 else 'se mantuvo similar'}:** El modelo ahora explica {final_r2*100:.2f}% de la varianza (vs {baseline_r2*100:.2f}% anterior).

    **Conclusión:** {' **¡Éxito!** El Grid Search encontró una configuración superior al modelo baseline.' if mejora_mae > 0 else '⚠️ El modelo baseline ya era muy bueno. Considera ampliar el espacio de búsqueda.'}
    ''' if mejora_mae > 0 else f'''
     **Nota:** El modelo baseline (Fase 2) sigue siendo competitivo. Esto puede significar que:
    1. Los hiperparámetros iniciales ya estaban bien elegidos
    2. El espacio de búsqueda del Grid Search no incluyó combinaciones mejores
    3. El modelo está cerca del límite de rendimiento para estos datos
    '''}
    """)


    fig_comparison, axes_comparison = plt.subplots(1, 3, figsize=(18, 5))

    metrics_names = ['R²', 'MAE', 'RMSE']
    colors_comparison = ['steelblue', 'forestgreen']

    for idx_m, (ax_comp, metric_name) in enumerate(zip(axes_comparison, metrics_names)):
        values_metric = comparison_df[metric_name].values
        bars_comp = ax_comp.bar(
            comparison_df['Modelo'], 
            values_metric, 
            color=colors_comparison, 
            alpha=0.85,
            edgecolor='black',
            linewidth=1.5
        )

        ax_comp.set_title(f'Comparación: {metric_name}', fontsize=14, fontweight='bold')
        ax_comp.set_ylabel(metric_name, fontsize=12)
        ax_comp.grid(axis='y', alpha=0.3, linestyle='--')
        ax_comp.set_axisbelow(True)

        # Añadir valores sobre las barras
        for bar_item in bars_comp:
            height_value = bar_item.get_height()
            ax_comp.text(
                bar_item.get_x() + bar_item.get_width()/2., 
                height_value,
                f'{height_value:.4f}' if metric_name == 'R²' else f'{height_value:.2f}',
                ha='center', 
                va='bottom', 
                fontweight='bold',
                fontsize=11
            )

        # Rotar etiquetas del eje x
        ax_comp.set_xticklabels(comparison_df['Modelo'], rotation=15, ha='right')

    plt.tight_layout()
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(" Comparación completada y guardada en:", comparison_plot_path)

    # Retornar variables
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Conclusiones y Recomendaciones

    ###  Logros de la Fase 3:

    1. **Grid Search Exitoso**: Evaluamos 108 combinaciones de hiperparámetros de forma sistemática
    2. **Optimización Documentada**: Todos los experimentos quedaron registrados en MLflow para trazabilidad
    3. **Modelo Mejorado**: Encontramos la configuración óptima que maximiza la precisión de predicción
    4. **Análisis Exhaustivo**: Identificamos patrones sobre qué hiperparámetros tienen mayor impacto

    ### Insights Descubiertos:

    - **Tipo de Capa**: {GRU/LSTM} demostró mejor rendimiento para este problema
    - **Tamaño Óptimo**: {X} unidades balancean capacidad y generalización
    - **Regularización**: Un dropout de {X}% previene efectivamente el overfitting
    - **Velocidad de Aprendizaje**: {X} permite convergencia estable

    ### Próximos Pasos Recomendados:

    1. **Ensemble Methods**: Combinar los top 5 modelos para predicciones más robustas
    2. **Optimización Bayesiana**: Explorar métodos más eficientes que Grid Search
    3. **Feature Engineering Avanzado**: Incorporar datos externos (días festivos, promociones)
    4. **Despliegue en Producción**: Crear API REST para servir predicciones en tiempo real
    5. **Monitoreo Continuo**: Implementar MLflow Model Registry para gestión de versiones

    ### Recursos para Profundizar:

    - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
    - [Keras Tuner](https://keras.io/keras_tuner/) (alternativa a Grid Search manual)
    - [Optuna](https://optuna.org/) (optimización bayesiana)

    ---

    ** Práctica completada exitosamente**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Acceso a MLflow UI

    Para visualizar todos los experimentos y comparar modelos:

    ```bash
    # En tu terminal, desde la carpeta del proyecto:
    mlflow ui
    ```

    Luego abre en tu navegador: `http://127.0.0.1:5000`

    ### Qué podrás ver en MLflow:

    - **Experiments**: Lista de todos los 108+ experimentos ejecutados
    - **Compare Runs**: Comparación visual de métricas entre modelos
    - **Artifacts**: Gráficos y modelos guardados
    - **Parameters**: Hiperparámetros de cada experimento
    - **Metrics**: Curvas de aprendizaje y métricas finales
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
