import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import joblib
    import os
    import mlflow               
    import mlflow.keras
    import math


    # ***** 

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import BatchNormalization
    return (
        Adam,
        Dense,
        Dropout,
        EarlyStopping,
        GRU,
        LabelEncoder,
        MinMaxScaler,
        ModelCheckpoint,
        Sequential,
        joblib,
        load_model,
        math,
        mean_absolute_error,
        mean_squared_error,
        mlflow,
        np,
        pd,
        plt,
        tf,
        zscore,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fase 01
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('../data/dataset_balanceado_500p_10r.csv')
    return (df,)


@app.cell
def _(df, pd):
    df["is_active"] = True

    # Transformar datos a tipo fecha
    df.created_at = pd.to_datetime(df.created_at)
    df.last_order_date = pd.to_datetime(df.last_order_date)
    df.last_updated_at = pd.to_datetime(df.last_updated_at)
    df.last_stock_count_date = pd.to_datetime(df.last_stock_count_date)
    df.expiration_date = pd.to_datetime(df.expiration_date)
    return


@app.cell
def _(df):
    df.head(25)
    return

@app.cell
def _(df, plt, sns, mo):
    
    # Título para la celda en Marimo
    mo.md("## Dashboard EDA: Balance, Realismo y Claridad del Dataset")

    # Configurar el estilo de los gráficos
    sns.set_style("whitegrid")

    # Crear una figura con 3 subplots (1 fila, 3 columnas)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Análisis de Estado del Dataset', fontsize=16, fontweight='bold')

    # --- 1. Gráfico de "Balance" ---
    # Contar registros por producto. Debería ser una línea plana en 10.
    conteo_productos = df['product_id'].value_counts()
    
    sns.histplot(conteo_productos, bins=1, ax=axes[0], color='blue')
    axes[0].set_title('1. Balance de Registros (¡Solucionado!)', fontsize=12)
    axes[0].set_xlabel('Registros por Producto')
    axes[0].set_ylabel('Cantidad de Productos')
    # Añadir texto de confirmación
    axes[0].text(0.5, 0.9, 'Todos los 500 productos\ntienen 10 registros', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=axes[0].transAxes, fontsize=12, color='green', fontweight='bold')

    # --- 2. Gráfico de "Realismo - Movimiento" ---
    # Usamos solo productos únicos para ver la distribución de "tipos" de producto
    df_unicos = df.drop_duplicates(subset=['product_id'])
    
    sns.histplot(df_unicos['average_daily_usage'], kde=True, ax=axes[1], color='orange', bins=50)
    axes[1].set_title('2. Realismo de Negocio (Movimiento)', fontsize=12)
    axes[1].set_xlabel('Uso Promedio Diario (Movimiento)')
    axes[1].set_ylabel('Frecuencia de Productos')
    # Limitar el eje X para ver mejor la distribución (quitando outliers extremos)
    axes[1].set_xlim(left=0, right=df_unicos['average_daily_usage'].quantile(0.95))

    # --- 3. Gráfico de "Realismo - Valor" ---
    sns.histplot(df_unicos['unit_cost'], kde=True, ax=axes[2], color='green', bins=50)
    axes[2].set_title('3. Realismo de Negocio (Valor)', fontsize=12)
    axes[2].set_xlabel('Costo Unitario (Valor)')
    axes[2].set_ylabel('Frecuencia de Productos')
    # Limitar el eje X
    axes[2].set_xlim(left=0, right=df_unicos['unit_cost'].quantile(0.95))

    # Ajustar y mostrar
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return fig, conteo_productos, df_unicos

@app.cell
def _(df, np, zscore):
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    # Iterar sobre cada columna para calcular outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not outliers.empty:
            print(f"\nColumna: '{col}'")
            print(f"Límites (IQR): ({lower_bound:.2f}, {upper_bound:.2f})")
            print(f"Total de outliers detectados: {len(outliers)}")
        else:
            print(f"\nColumna: '{col}' -> Sin outliers (según IQR).")


    threshold = 3 

    for col_2 in numeric_cols:
        z_scores = np.abs(zscore(df[col_2]))

        outliers_2 = df[z_scores > threshold]

        if not outliers.empty:
            print(f"\nColumna: '{col_2}'")
            print(f"Umbral (Z-score): {threshold}")
            print(f"Total de outliers detectados: {len(outliers_2)}")
            # print(outliers[[col_2, 'product_name']].sort_values(by=col_2, ascending=False).head())
        else:
            print(f"\nColumna: '{col_2}' -> Sin outliers (Z-score < {threshold}).")
    return


@app.cell
def _(df):
    print("Iniciando Feature Engineering...")

    df_feat = df.copy()

    # Usando 'created_at' como la fecha principal del registro
    base_date = df_feat['created_at']

    df_feat['dia_del_mes'] = base_date.dt.day
    df_feat['dia_de_la_semana'] = base_date.dt.dayofweek # Lunes=0, Domingo=6
    df_feat['mes'] = base_date.dt.month
    df_feat['trimestre'] = base_date.dt.quarter
    df_feat['es_fin_de_semana'] = df_feat['dia_de_la_semana'].isin([5, 6]).astype(int)

    print("Variables temporales creadas.")

    # 1. Días restantes hasta vencimiento
    df_feat['dias_para_vencimiento'] = (df_feat['expiration_date'] - base_date).dt.days
    # Manejar valores negativos (si 'created_at' es posterior a 'expiration_date')
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].fillna(0)
    df_feat['dias_para_vencimiento'] = df_feat['dias_para_vencimiento'].apply(lambda x: max(0, x))

    # 2. Antigüedad del producto (Sugerido en la guía)
    df_feat['antiguedad_producto_dias'] = (base_date - df_feat['last_stock_count_date']).dt.days
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].fillna(0)
    df_feat['antiguedad_producto_dias'] = df_feat['antiguedad_producto_dias'].apply(lambda x: max(0, x))


    # 3. Ratio de uso sobre stock (Sugerido en la guía)
    df_feat['ratio_uso_stock'] = df_feat['average_daily_usage'] / (df_feat['quantity_available'] + 1)

    # Mostramos las columnas clave y las nuevas que creamos
    columnas_a_mostrar = [
        'created_at', 
        'product_id', 
        'quantity_available', 
        'average_daily_usage',
        'expiration_date',
        # --- Nuevas ---
        'dia_de_la_semana', 
        'mes', 
        'es_fin_de_semana',
        'dias_para_vencimiento',
        'antiguedad_producto_dias',
        'ratio_uso_stock'
    ]

    print("Creando feature 'necesita_reorden'...")
    df_feat['necesita_reorden'] = (df_feat['quantity_on_hand'] <= df_feat['reorder_point']).astype(int)

    print(df_feat[columnas_a_mostrar].head())
    print(df_feat[columnas_a_mostrar].info())
    return (df_feat,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Aplicación de One-Hot Encoding
    """)
    return


@app.cell
def _(LabelEncoder, df_feat, joblib, pd):
    df_proc = df_feat.copy()

    # Codificador para product_id
    le_product_id = LabelEncoder()
    df_proc['product_id_encoded'] = le_product_id.fit_transform(df_proc['product_id'])
    joblib.dump(le_product_id, 'le_product_id.joblib') # Guardar

    # Codificador para supplier_id
    le_supplier_id = LabelEncoder()
    df_proc['supplier_id_encoded'] = le_supplier_id.fit_transform(df_proc['supplier_id'])
    joblib.dump(le_supplier_id, 'le_supplier_id.joblib') # Guardar

    categorias_onehot = ['warehouse_location', 'stock_status']
    df_proc = pd.get_dummies(df_proc, columns=categorias_onehot, drop_first=True)

    print("\nColumnas después de One-Hot Encoding:")
    print([col for col in df_proc.columns if 'warehouse_location_' in col or 'stock_status_' in col])
    return (df_proc,)


@app.cell
def _(MinMaxScaler, df_proc, joblib):
    #Se añadio la cariable reorden
    columnas_numericas = [
        'quantity_on_hand', 'quantity_reserved', 'quantity_available',
        'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
        'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
        'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana',
        'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock', 
        'necesita_reorden'
    ]

    scaler = MinMaxScaler()
    df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])
    joblib.dump(scaler, 'min_max_scaler.joblib')

    print("\nEscalar Variables Numéricas")
    return


@app.cell
def _(df_proc):
    print("\nDataFrame Procesado")
    print(df_proc.head())

    columnas_modelo = df_proc.select_dtypes(exclude=['object', 'datetime64[ns]']).columns
    print(df_proc[columnas_modelo].info())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ¿Por qué el dataset final tiene 48 columnas?

    1. El dataset original (con 28 columnas) no era el mejor para el modelo. Un modelo de Deep Learning no puede entender texto (`object`) ni fechas (`datetime`).

    2. Las 20 columnas adicionales son el resultado de la Ingeniería de Características (Feature Engineering), en este paso convertimos todo a números:

    3. 7 Columnas de Fechas: Usamos las fechas para crear variables numéricas como `dia_del_mes`, `mes`, y `dias_para_vencimiento`.
    4. 2 Columnas de Negocio: Creamos ratios como `ratio_uso_stock` y `necesita_reorden` para dar más contexto al modelo.
    5. 11 Columnas de Codificación:
    6. `product_id` y `supplier_id` (texto) se convirtieron en `_encoded` (números).
    7. `warehouse_location` y `stock_status` (texto) se convirtieron en múltiples columnas de 0s y 1s (One-Hot Encoding).

    - El resultado es un dataset de 48 columnas puramente numérico, que es exactamente lo que el modelo necesita para entrenar.
    """)
    return


@app.cell
def _(df_feat, df_proc):
    df_proc['created_at'] = df_feat['created_at']
    print("DataFrame 'df_proc' listo para la creación de secuencias.")
    return


@app.cell
def _(df_proc):
    # 7 días.
    N_STEPS = 4

    TARGET_COLUMN = 'quantity_available'

    FEATURE_COLUMNS = [
        'quantity_on_hand', 'quantity_reserved',  # variables base del stock
        'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
        'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value',
        'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre',
        'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias',
        'ratio_uso_stock', 'necesita_reorden',
        'product_id_encoded', 'supplier_id_encoded',
        'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte',
        'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur',
        'warehouse_location_Centro Distribución 1',
        'warehouse_location_Centro Distribución 2',
        'stock_status_1', 'stock_status_2', 'stock_status_3'
    ]


    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_proc.columns]
    if missing_cols:
        print(f"Faltan las columnas: {missing_cols}")
    else:
        print(f"Todas las {len(FEATURE_COLUMNS)} features están presentes.")

    if TARGET_COLUMN not in FEATURE_COLUMNS:
        print(f"Target '{TARGET_COLUMN}' no en features.")
    return FEATURE_COLUMNS, N_STEPS, TARGET_COLUMN


@app.cell
def _(df_proc, np):

    print("\nDividiendo en Train y Validation (80/20 split por PRODUCTO)")

    # 1. Obtener todos los IDs de productos únicos
    all_product_ids = df_proc['product_id_encoded'].unique()

    # 2. Barajarlos aleatoriamente para que la división sea justa
    np.random.seed(42) # Para que la división sea siempre la misma
    np.random.shuffle(all_product_ids)

    # 3. Definir el punto de corte (80% de los productos)
    split_index = int(len(all_product_ids) * 0.8)

    # 4. Dividir los IDs de los productos
    train_product_ids = all_product_ids[:split_index]
    val_product_ids = all_product_ids[split_index:]

    # 5. Crear los DataFrames de train y val usando los IDs
    # train_df contendrá TODAS las filas (los 10 días) de los 400 productos
    train_df = df_proc[df_proc['product_id_encoded'].isin(train_product_ids)]

    # val_df contendrá TODAS las filas (los 10 días) de los 100 productos
    val_df = df_proc[df_proc['product_id_encoded'].isin(val_product_ids)]

    print(f"Total de productos: {len(all_product_ids)}")
    print(f"Productos de Entrenamiento (Train): {len(train_product_ids)}")
    print(f"Productos de Validación (Val): {len(val_product_ids)}")
    print(f"Registros de Entrenamiento (Train): {len(train_df)}")
    print(f"Registros de Validación (Val): {len(val_df)}")

    # Verificación de que no se mezclen
    assert len(set(train_product_ids) & set(val_product_ids)) == 0, "Error: ¡Productos duplicados en train y val!"

    return train_df, val_df


@app.cell
def _(np):
    def create_sequences(data_df, product_group, n_steps, feature_cols, target_col):
        product_data = data_df[data_df['product_id_encoded'] == product_group].copy()

        product_data = product_data.sort_values(by='created_at')

        features = product_data[feature_cols].values
        target = product_data[target_col].values

        X, y = [], []

        for i in range(n_steps, len(product_data)):
            X.append(features[i-n_steps:i])

            y.append(target[i])

        if len(X) > 0:
            return np.array(X), np.array(y)
        else:
            return None, None
    return (create_sequences,)


@app.cell
def _(
    FEATURE_COLUMNS,
    N_STEPS,
    TARGET_COLUMN,
    create_sequences,
    np,
    train_df,
    val_df,
):
    print('\nProcesando secuencias para Train y Validation')
    X_train_list, y_train_list = ([], [])
    X_val_list, y_val_list = ([], [])
    print('procesar set entrenamiento')
    unique_products_train = train_df['product_id_encoded'].unique()
    for _product_id in unique_products_train:
        X_prod, y_prod = create_sequences(train_df, _product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)
        if X_prod is not None:
            X_train_list.append(X_prod)
            y_train_list.append(y_prod)
    print('procesar set validacion...')
    unique_products_val = val_df['product_id_encoded'].unique()
    for _product_id in unique_products_val:
        X_prod, y_prod = create_sequences(val_df, _product_id, N_STEPS, FEATURE_COLUMNS, TARGET_COLUMN)
        if X_prod is not None:
            X_val_list.append(X_prod)
            y_val_list.append(y_prod)
    if len(X_train_list) > 0:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        print(f'Forma de X_train (Muestras, Pasos, Features): {X_train.shape}')
        print(f'Forma de y_train (Muestras,): {y_train.shape}')
        print(f'Forma de X_val (Muestras, Pasos, Features): {X_val.shape}')
        print(f'Forma de y_val (Muestras,): {y_val.shape}')
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('y_val.npy', y_val)
    else:
        print('\nNo hay secuencias.')
    return


@app.cell
def _(mo):
    mo.md(r"""
    La forma correcta de hacerlo es dividir por producto, no por fila.

    Entrenamiento: 400 productos (con sus 10 días de historia completos).

    Validación: 100 productos (con sus 10 días de historia completos).

    Esto es una prueba mucho más realista: el modelo entrenará con 400 productos y lo probaremos en 100 productos completamente nuevos que nunca ha visto.
    """)
    return


@app.cell
def _(df_proc, pd):
    df_proc_path = 'df_processed_features.csv'
    df_proc['created_at'] = pd.to_datetime(df_proc['created_at'])
    df_proc.to_csv(df_proc_path, index=False)
    print(f"DataFrame procesado guardado en '{df_proc_path}'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explicacion de uso dataset de 500 productos

    Dataset correcto:
        - dataset_balanceado_500p_10r.csv

        Por qué:

        - Ese dataset ya resolvió el desbalance malo (cantidad de filas por producto).
            Ahora todos los productos tienen las mismas oportunidades de aprendizaje.

        - Mantiene el desbalance bueno, es decir, la diversidad natural del negocio:

            1. Algunos productos se venden más.

            2. Otros cuestan más.

            3. Otros rotan menos.


    Importante:  No se suavizó ni normalizó las diferencias de valores entre productos (como average_daily_usage o unit_cost).
    Esas diferencias son la información que el modelo necesita para entender la lógica del negocio.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fase 02
    """)
    return


@app.cell
def _(np, tf):
    np.random.seed(42)
    tf.random.set_seed(42)
    PATH_X_TRAIN = 'X_train.npy'
    PATH_Y_TRAIN = 'y_train.npy'
    PATH_X_VAL = 'X_val.npy'
    PATH_Y_VAL = 'y_val.npy'
    X_train_1 = np.load(PATH_X_TRAIN, allow_pickle=True)
    y_train_1 = np.load(PATH_Y_TRAIN, allow_pickle=True)
    X_val_1 = np.load(PATH_X_VAL, allow_pickle=True)
    y_val_1 = np.load(PATH_Y_VAL, allow_pickle=True)
    print("\nConvirtiendo arrays a dtype 'float32'...")
    X_train_1 = X_train_1.astype('float32')
    y_train_1 = y_train_1.astype('float32')
    # Convertir arrays de 'object' a 'float32' para TensorFlow
    X_val_1 = X_val_1.astype('float32')
    y_val_1 = y_val_1.astype('float32')
    print('\n--- 2. Verificación de Formas (Shapes) ---')
    print(f'Forma de X_train (Muestras, Pasos, Features): {X_train_1.shape}')
    print(f'Forma de y_train (Muestras,): {y_train_1.shape}')
    print(f'Forma de X_val (Muestras, Pasos, Features): {X_val_1.shape}')
    print(f'Forma de y_val (Muestras,): {y_val_1.shape}')
    INPUT_SHAPE = (X_train_1.shape[1], X_train_1.shape[2])
    print('Datos cargados')
    return INPUT_SHAPE, X_train_1, X_val_1, y_train_1, y_val_1


@app.cell
def _(Dense, Dropout, GRU, INPUT_SHAPE, Sequential):

    # --- Hiperparámetros ---
    HP_GRU_UNITS = 32  # <-- ¡MUCHO MÁS SIMPLE! (Antes 128+64)
    HP_DROPOUT = 0.4   # <-- ¡MÁS ALTO! (Antes 0.3)
    HP_LR = 0.0005     # <-- Tu Learning Rate

    # Arquitectura SIMPLE para evitar overfitting
    model = Sequential(name="Modelo_GRU_Simple")
    model.add(GRU(HP_GRU_UNITS, input_shape=INPUT_SHAPE, name="GRU_1"))
    model.add(Dropout(HP_DROPOUT, name="Dropout_1"))
    model.add(Dense(1, activation='sigmoid', name="Salida_Prediccion"))

    model.summary()

    # ¡Devuelve los HPs para que MLflow los vea!
    return HP_DROPOUT, HP_GRU_UNITS, HP_LR, model


@app.cell
def _(Adam, EarlyStopping, HP_LR, ModelCheckpoint, model):

    model.compile(
        optimizer=Adam(learning_rate=HP_LR), # <--- Usa la variable
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )
    print("Compilado")


    # Callbacks

    # Esto es para guardar solo el modelo que tenga el val_loss más bajo.
    checkpoint_path = 'best_model.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss', 
        save_best_only=True,
        mode='min',
        verbose=1 
    )

    # Detener el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # esperar 15 épocas sin mejora
        mode='min',
        verbose=1,
    )

    print("Callbacks")
    return early_stopping, model_checkpoint


@app.cell
def _(
    HP_DROPOUT,
    HP_GRU_UNITS,
    HP_LR,
    X_train_1,
    X_val_1,
    early_stopping,
    mlflow,
    model,
    model_checkpoint,
    y_train_1,
    y_val_1,
):

    EPOCHS = 100
    BATCH_SIZE = 64

    # Iniciar experimento MLflow
    mlflow.set_experiment("Optimizacion_Stock_Practica2")

    with mlflow.start_run() as run:
        # --- ¡CORRECCIÓN! ---
        # Ahora registramos los HPs reales que usó el modelo
        mlflow.log_param("gru_units", HP_GRU_UNITS)
        mlflow.log_param("dropout", HP_DROPOUT)
        mlflow.log_param("learning_rate", HP_LR)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # Habilitar autologging
        mlflow.keras.autolog()

        # Entrenamiento
        history = model.fit(
            X_train_1, y_train_1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_1, y_val_1),
            callbacks=[model_checkpoint, early_stopping],
            verbose=1
        )

    print(f"Entrenamiento completo con MLflow. Run ID: {run.info.run_id}")
    return (history,)


@app.cell
def _(history, plt):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    epochs_range = range(len(loss)) # El número de épocas que realmente corrió

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

    plt.show()
    return


@app.cell
def _(
    X_val_1,
    load_model,
    math,
    mean_absolute_error,
    mean_squared_error,
    y_val_1,
):
    best_model = load_model('best_model.keras')
    y_pred_scaled = best_model.predict(X_val_1)
    # Predicciones
    rmse_scaled = math.sqrt(mean_squared_error(y_val_1, y_pred_scaled))
    mae_scaled = mean_absolute_error(y_val_1, y_pred_scaled)
    # Métricas
    print(f'Métricas del Modelo (en datos escalados [0, 1]):')
    print(f'RMSE: {rmse_scaled:.4f}')
    print(f'MAE:  {mae_scaled:.4f}')
    return mae_scaled, rmse_scaled, y_pred_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Esto indica que el error promedio y cuadrático medio representan apenas el 5–6 % del rango total de los datos normalizados, lo que es muy bajo y evidencia una alta precisión del modelo.
    """)
    return


@app.cell
def _(
    joblib,
    mae_scaled,
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    rmse_scaled,
    y_pred_scaled,
    y_val_1,
):
    # ---  Desescalar valores y calcular métricas en unidades reales ---
    scaler_1 = joblib.load('min_max_scaler.joblib')
    columnas_numericas_1 = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level', 'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock', 'necesita_reorden']
    TARGET_COLUMN_1 = 'quantity_available'
    TARGET_COLUMN_INDEX = columnas_numericas_1.index(TARGET_COLUMN_1)
    num_numeric_features = len(columnas_numericas_1)
    print(f' TARGET_COLUMN_INDEX = {TARGET_COLUMN_INDEX}')
    # 1️ Cargar el scaler entrenado
    print(f' Total de columnas numéricas = {num_numeric_features}')
    dummy_y_val = np.zeros((len(y_val_1), num_numeric_features))
    # 2️Lista original usada en el escalado (debe coincidir con la de tu pipeline)
    dummy_y_val[:, TARGET_COLUMN_INDEX] = y_val_1.ravel()
    y_val_real = scaler_1.inverse_transform(dummy_y_val)[:, TARGET_COLUMN_INDEX]
    dummy_y_pred = np.zeros((len(y_pred_scaled), num_numeric_features))
    dummy_y_pred[:, TARGET_COLUMN_INDEX] = y_pred_scaled.ravel()
    y_pred_real = scaler_1.inverse_transform(dummy_y_pred)[:, TARGET_COLUMN_INDEX]
    rmse_real = math.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mae_real = mean_absolute_error(y_val_real, y_pred_real)
    min_stock = scaler_1.data_min_[TARGET_COLUMN_INDEX]
    max_stock = scaler_1.data_max_[TARGET_COLUMN_INDEX]
    # 3️ Detectar dinámicamente la columna objetivo
    rango_stock = max_stock - min_stock
    error_relativo = mae_real / rango_stock * 100
    print('\n MÉTRICAS FINALES DEL MODELO')
    print(f'   • Rango de stock: {min_stock:.0f} - {max_stock:.0f} unidades')
    print(f'   • Rango total: {rango_stock:.0f} unidades\n')
    print(f'   • RMSE (real): {rmse_real:.2f} unidades')
    print(f'   • MAE  (real): {mae_real:.2f} unidades')
    # 4️ Reconstruir arrays dummy para invertir la escala
    print(f'   • Error Relativo: {error_relativo:.2f}%')
    if error_relativo <= 10:
        calidad = ' Excelente'
    elif error_relativo <= 20:
        calidad = ' Aceptable'
    elif error_relativo <= 30:
        calidad = ' Regular'
    else:
    # 5️ Calcular métricas reales
        calidad = ' Deficiente'
    print(f'\nCalidad del modelo según error relativo: {calidad}')
    # 6️ Contexto del rango real del target
    # 7️ Reporte final
    # Clasificación rápida de desempeño
    print(f'   (RMSE normalizado: {rmse_scaled:.4f}, MAE normalizado: {mae_scaled:.4f})')
    return (
        columnas_numericas_1,
        error_relativo,
        mae_real,
        rmse_real,
        y_pred_real,
        y_val_real,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Un error relativo menor al 10 % se considera excelente desempeño en modelos de regresión de inventario o series continuas.
    El modelo tiene una desviación típica de solo ~340 unidades frente a un rango de 5238, lo que significa que el modelo generaliza bien sin sobreajustar.
    """)
    return


@app.cell
def _(plt, y_pred_real, y_val_real):
    plt.figure(figsize=(14, 6))

    # Scatter plot de predicciones vs reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_val_real, y_pred_real, alpha=0.3, s=10, edgecolors='none', color='steelblue')

    # Línea diagonal perfecta
    min_val = min(y_val_real.min(), y_pred_real.min())
    max_val = max(y_val_real.max(), y_pred_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')

    plt.xlabel('Valor Real (unidades)', fontsize=11)
    plt.ylabel('Valor Predicho (unidades)', fontsize=11)
    plt.title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histograma comparativo
    plt.subplot(1, 2, 2)
    plt.hist(y_val_real, bins=50, alpha=0.5, label='Valores Reales', color='blue', edgecolor='black')
    plt.hist(y_pred_real, bins=50, alpha=0.5, label='Predicciones', color='orange', edgecolor='black')
    plt.xlabel('Stock (unidades)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución: Real vs Predicho', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La nube de puntos sigue de cerca la línea roja de predicción perfecta, sin dispersión excesiva en los extremos → predicciones consistentes en todo el rango de valores, no solo en el rango medio.
    """)
    return


@app.cell
def _(np, plt, y_pred_real, y_val_real):
    errors = y_pred_real - y_val_real
    abs_errors = np.abs(errors)
    percent_errors = (abs_errors / (y_val_real + 1)) * 100

    plt.figure(figsize=(15, 5))

    # Subplot 1: Histograma de errores (con signo)
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    plt.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Media = {np.mean(errors):.2f}')
    plt.xlabel('Error (Predicción - Real)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución de Errores', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Histograma de errores absolutos
    plt.subplot(1, 3, 2)
    plt.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(abs_errors), color='darkred', linestyle='--', linewidth=2, 
                label=f'MAE = {np.mean(abs_errors):.2f}')
    plt.xlabel('Error Absoluto (unidades)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title('Distribución de Errores Absolutos', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Boxplot de errores absolutos
    plt.subplot(1, 3, 3)
    box = plt.boxplot(abs_errors, vert=True, patch_artist=True, 
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    plt.ylabel('Error Absoluto (unidades)', fontsize=11)
    plt.title('Boxplot de Errores Absolutos', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Añadir estadísticas
    q1, median, q3 = np.percentile(abs_errors, [25, 50, 75])
    plt.text(1.15, median, f'Mediana: {median:.1f}', fontsize=9, va='center')
    plt.text(1.15, q1, f'Q1: {q1:.1f}', fontsize=9, va='center')
    plt.text(1.15, q3, f'Q3: {q3:.1f}', fontsize=9, va='center')

    plt.tight_layout()
    plt.show()
    return abs_errors, percent_errors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Los errores se centran cerca de 0, con media ≈ 127 y pocos valores extremos.

    Los errores absolutos tienen una forma aproximadamente normal y el boxplot muestra algunos outliers esperables.
        - Esto significa que no hay sesgo sistemático (el modelo no sobrestima ni subestima de forma consistente).
    """)
    return


@app.cell
def _(abs_errors, np, plt, y_val_real):
    percentiles = [0, 33, 66, 100]
    bins = np.percentile(y_val_real, percentiles)

    stock_ranges = ['Bajo (0-33%)', 'Medio (33-66%)', 'Alto (66-100%)']
    range_indices = [
        (y_val_real >= bins[0]) & (y_val_real < bins[1]),
        (y_val_real >= bins[1]) & (y_val_real < bins[2]),
        (y_val_real >= bins[2])
    ]

    # Calcular métricas por rango
    print("ANÁLISIS DE RENDIMIENTO POR RANGO DE STOCK")

    range_stats = []
    for i, (range_name, indices) in enumerate(zip(stock_ranges, range_indices)):
        range_errors = abs_errors[indices]
        range_vals = y_val_real[indices]

        mae_range = np.mean(range_errors)
        count = indices.sum()
        pct = (count / len(y_val_real)) * 100

        range_stats.append({
            'name': range_name,
            'count': count,
            'percentage': pct,
            'mae': mae_range,
            'median_error': np.median(range_errors),
            'std_error': range_errors.std(),
            'min_stock': range_vals.min(),
            'max_stock': range_vals.max()
        })

        print(f"\n{range_name}:")
        print(f"   • Rango: [{range_vals.min():.0f} - {range_vals.max():.0f}] unidades")
        print(f"   • Cantidad de muestras: {count:,} ({pct:.1f}%)")
        print(f"   • MAE: {mae_range:.2f} unidades")
        print(f"   • Error mediano: {np.median(range_errors):.2f} unidades")
        print(f"   • Desviación estándar: {range_errors.std():.2f} unidades")


    # Visualización por rangos
    plt.figure(figsize=(15, 5))

    # Gráfico 1: MAE por rango
    plt.subplot(1, 3, 1)
    maes = [stat['mae'] for stat in range_stats]
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    bars = plt.bar(stock_ranges, maes, color=colors, edgecolor='black', alpha=0.7)
    plt.ylabel('MAE (unidades)', fontsize=11)
    plt.title('Error Absoluto Medio por Rango', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, mae_iter in zip(bars, maes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_iter:.1f}', ha='center', va='bottom', fontweight='bold')

    # Gráfico 2: Distribución de muestras
    plt.subplot(1, 3, 2)
    counts = [stat['count'] for stat in range_stats]
    plt.pie(counts, labels=stock_ranges, autopct='%1.1f%%', colors=colors, startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    plt.title('Distribución de Muestras por Rango', fontsize=12, fontweight='bold')

    # Gráfico 3: Boxplot comparativo
    plt.subplot(1, 3, 3)
    error_data = [abs_errors[indices] for indices in range_indices]
    box_i = plt.boxplot(error_data, labels=['Bajo', 'Medio', 'Alto'], 
                      patch_artist=True, notch=True)

    for patch, color in zip(box_i['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('Error Absoluto (unidades)', fontsize=11)
    plt.xlabel('Rango de Stock', fontsize=11)
    plt.title('Distribución de Errores por Rango', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return (range_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El modelo predice mejor los stocks altos, lo que es lógico porque hay más información y menor ruido en esos valores.
    La desviación estándar en todos los rangos es moderada (≈200 unidades), lo que muestra consistencia.
    """)
    return


@app.cell
def _(error_relativo, mae_real, percent_errors, range_stats, rmse_real):
    print("ANÁLISIS DE RENDIMIENTO DEL MODELO")

    print("\nMÉTRICAS GLOBALES:")
    print(f"   • MAE: {mae_real:.2f} unidades ({error_relativo:.2f}% del rango)")
    print(f"   • RMSE: {rmse_real:.2f} unidades")
    print(f"   • Ratio RMSE/MAE: {rmse_real/mae_real:.2f}")

    print("\nDISTRIBUCIÓN DE CALIDAD:")
    excellent = (percent_errors < 5).sum()
    good = ((percent_errors >= 5) & (percent_errors < 10)).sum()
    fair = ((percent_errors >= 10) & (percent_errors < 20)).sum()
    poor = (percent_errors >= 20).sum()
    total = len(percent_errors)

    print(f"   • Excelente (<5% error):  {excellent:,} predicciones ({excellent/total*100:.1f}%)")
    print(f"   • Bueno (5-10% error):    {good:,} predicciones ({good/total*100:.1f}%)")
    print(f"   • Aceptable (10-20%):     {fair:,} predicciones ({fair/total*100:.1f}%)")
    print(f"   • Necesita mejora (>20%): {poor:,} predicciones ({poor/total*100:.1f}%)")

    print("\nRENDIMIENTO POR RANGO DE STOCK:")
    for stat in range_stats:
        print(f"   • {stat['name']:15} → MAE: {stat['mae']:6.2f} unidades ({stat['percentage']:5.1f}% de datos)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusiones de la fase 2 de entrenamiento
    El modelo presenta un error relativo del 5.14 %, lo que indica una excelente capacidad predictiva en el rango completo de stock. Las métricas escaladas (RMSE=0.065, MAE=0.051) confirman una buena normalización del error. Además, las gráficas muestran una correlación casi lineal entre los valores reales y predichos, evidenciando que el modelo generaliza correctamente sin sobreajuste. El rendimiento es especialmente estable en los rangos medio y alto, con menor dispersión en los errores absolutos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fase 03
    """)
    return


@app.cell
def _(joblib, load_model, pd):
    N_STEPS_1 = 4
    TARGET_COLUMN_INDEX_1 = 2
    NUM_NUMERIC_FEATURES = 18
    try:
    # modelo
        model_1 = load_model('best_model.keras')
        print("Modelo 'best_model.keras' cahrgado.")
    except Exception as e:
        print(f"Error al cargar 'best_model.keras': {e}")
    try:
        scaler_2 = joblib.load('min_max_scaler.joblib')
    # escalador
        print("Escalador 'min_max_scaler.joblib' cargado.")
    except Exception as e:
        print(f"Error al cargar 'min_max_scaler.joblib': {e}")
    try:
        le_product_id_1 = joblib.load('le_product_id.joblib')
        print("Codificador 'le_product_id.joblib' cargado.")
    # codificador de productos
    except Exception as e:
        print(f"Error al cargar 'le_product_id.joblib': {e}")
    try:
        df_features = pd.read_csv('df_processed_features.csv')
        df_features['created_at'] = pd.to_datetime(df_features['created_at'])
        print(f'Base de datos de features cargada ({len(df_features)} registros).')
    # features
    except Exception as e:
        print(f"Error al cargar 'df_processed_features.csv': {e}")
    # Lista de columnas
    FEATURE_COLUMNS_1 = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level', 'reorder_quantity', 'average_daily_usage', 'unit_cost', 'total_value', 'is_active', 'dia_del_mes', 'dia_de_la_semana', 'mes', 'trimestre', 'es_fin_de_semana', 'dias_para_vencimiento', 'antiguedad_producto_dias', 'ratio_uso_stock', 'product_id_encoded', 'supplier_id_encoded', 'warehouse_location_Almacén Este', 'warehouse_location_Almacén Norte', 'warehouse_location_Almacén Oeste', 'warehouse_location_Almacén Sur', 'warehouse_location_Centro Distribución 1', 'warehouse_location_Centro Distribución 2', 'stock_status_1', 'stock_status_2', 'stock_status_3']
    return (
        FEATURE_COLUMNS_1,
        N_STEPS_1,
        TARGET_COLUMN_INDEX_1,
        df_features,
        le_product_id_1,
        model_1,
        scaler_2,
    )


@app.cell
def _(
    FEATURE_COLUMNS_1,
    N_STEPS_1,
    TARGET_COLUMN_INDEX_1,
    columnas_numericas_1,
    df_features,
    le_product_id_1,
    model_1,
    np,
    pd,
    scaler_2,
):
    def predict_demand(product_id_str, target_date_str):
        try:  # Validar y codificar ID de producto
            product_id_encoded = le_product_id_1.transform([product_id_str])[0]
        except ValueError:
            return f"Error: El ID de producto '{product_id_str}' no fue visto durante el entrenamiento."
        try:
            target_date = pd.to_datetime(target_date_str)
        except ValueError:  # Validar formato de fecha
            return f"Error: Formato de fecha incorrecto '{target_date_str}'."
        product_data = df_features[df_features['product_id_encoded'] == product_id_encoded].sort_values(by='created_at')
        historical_data = product_data[product_data['created_at'] < target_date]
        if len(historical_data) < N_STEPS_1:
            return f'Error: No hay suficiente historia ({len(historical_data)} días). Se necesitan {N_STEPS_1} días.'
        sequence_df = historical_data.tail(N_STEPS_1)  # Extraer datos históricos
        input_features_df = sequence_df[FEATURE_COLUMNS_1]
        input_features_scaled = input_features_df.astype(np.float32).values
        input_sequence = np.expand_dims(input_features_scaled, axis=0)
        try:  # Validar historia mínima
            pred_scaled = model_1.predict(input_sequence, verbose=0)
            pred_scaled = float(pred_scaled.ravel()[0])
        except Exception as e:
            return f'Error en la predicción: {e}'  # Preparar secuencia de entrada
        dummy_pred = np.zeros((1, len(columnas_numericas_1)), dtype=np.float32)
        dummy_pred[0, TARGET_COLUMN_INDEX_1] = pred_scaled
        pred_real = scaler_2.inverse_transform(dummy_pred)[0][TARGET_COLUMN_INDEX_1]
        print('Predicción realizada')
        return max(0, pred_real)  # Predicción  # Asegura que sea un escalar  # Desescalar la predicción
    return (predict_demand,)


@app.cell
def _(np, pd, predict_demand):
    df_original = pd.read_csv('../data/dataset_balanceado_500p_10r.csv')
    unique_products = df_original['product_id'].unique()
    NUM_PRODUCTS = 20
    TARGET_DATE = '2025-10-31'
    np.random.seed(42)
    sample_products = np.random.choice(unique_products, size=min(NUM_PRODUCTS, len(unique_products)), replace=False)
    print(f'PREDICCIONES PARA {len(sample_products)} PRODUCTOS ÚNICOS')
    print(f'Fecha objetivo: {TARGET_DATE}')
    results = []
    success_count = 0
    for idx, _product_id in enumerate(sample_products, 1):
        print(f'\n[{idx}/{len(sample_products)}] {_product_id}')
        prediction = predict_demand(_product_id, TARGET_DATE)
        if isinstance(prediction, (int, float)):
            success_count = success_count + 1
            print(f'Stock predicho: {prediction:.2f} unidades')
            results.append({'product_id': _product_id, 'prediction': prediction, 'status': 'success'})
        else:
            print(f'{prediction}')
            results.append({'product_id': _product_id, 'prediction': None, 'status': 'failed'})
    results_df = pd.DataFrame(results)
    if success_count > 0:
        successful = results_df[results_df['status'] == 'success']['prediction']
        print(f'\nEstadísticas:')
        print(f'   Media: {successful.mean():.2f} unidades')
        print(f'   Mediana: {successful.median():.2f} unidades')
        print(f'   Mínimo: {successful.min():.2f} unidades')
        print(f'   Máximo: {successful.max():.2f} unidades')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
