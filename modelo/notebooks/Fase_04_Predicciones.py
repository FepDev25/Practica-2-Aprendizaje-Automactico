import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 4: Predicciones con Nuevos Datos
    
    ##  Objetivo
    
    En esta fase vamos a:
    1. Cargar el **mejor modelo** entrenado (de Fase 2 o Fase 3)
    2. Cargar los **scalers** (para normalizar nuevos datos)
    3. Procesar **nuevos datos** con el mismo pipeline de Fase 1
    4. Hacer **predicciones** y analizarlas
    
    ##  Requisitos
    
    - Modelo guardado: `model.keras` (de Fase 2)
    - Scalers guardados: `scaler_X.pkl` y `scaler_y.pkl`
    - Nuevo dataset: `nuevos_datos.csv`
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import pickle
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(" Librerías importadas")
    return load_model, np, pd, pickle, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Cargar Modelo y Scalers
    
    Cargamos el modelo entrenado y los objetos MinMaxScaler que usamos
    en la Fase 2 para normalizar los datos.
    """)
    return


@app.cell
def _(load_model, mo, pickle):
    # Cargar modelo
    try:
        modelo_produccion = load_model('model.keras')
        print(" Modelo cargado desde 'model.keras'")
    except FileNotFoundError:
        mo.stop(True, mo.md(" No se encontró 'model.keras'. Ejecuta Fase_02.py primero."))
    
    # Cargar scalers (primero debes guardarlos en Fase_02)
    try:
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X_prod = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y_prod = pickle.load(f)
        print(" Scalers cargados correctamente")
    except FileNotFoundError:
        mo.stop(True, mo.md("""
        
                             No se encontraron los scalers. 
        
        **Solución:** Añade esto al final de Fase_02.py:
        ```python
        import pickle
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open('scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        ```
        """))
    
    return modelo_produccion, scaler_X_prod, scaler_y_prod


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Cargar y Procesar Nuevos Datos
    
    **Importante:** Los nuevos datos deben tener:
    - Las mismas 38 features que usamos en el entrenamiento
    - El mismo procesamiento (lags, medias móviles, etc.)
    
    **Opciones:**
    - **Opción A:** Usar el mismo dataset pero simular "nuevos datos"
    - **Opción B:** Cargar un CSV completamente nuevo
    """)
    return


@app.cell
def _(pd):
    # Opción A: Simular nuevos datos (usando los últimos 100 registros del dataset original)
    nuevos_datos = pd.read_csv(
        "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_processed_advanced.csv"
    ).tail(100)  # Tomar últimos 100 registros
    
    # Eliminar columnas no numéricas
    nuevos_datos = nuevos_datos.drop(columns=['product_sku', 'region_almacen'], errors='ignore')
    
    print(f" Nuevos datos cargados: {nuevos_datos.shape}")
    nuevos_datos.head()
    return (nuevos_datos,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Preparar Datos para Predicción
    
    Aplicamos el mismo proceso que en Fase_02:
    1. Separar X e y (aunque no usaremos y para predicción)
    2. Escalar con los **mismos scalers** del entrenamiento
    3. Crear secuencias de 7 días
    """)
    return


@app.cell
def _(nuevos_datos, np, scaler_X_prod):
    # 1. Separar features
    y_nuevos = nuevos_datos[['quantity_available']]  # Solo para comparar después
    X_nuevos = nuevos_datos.drop(columns=['quantity_available'])
    
    # 2. Escalar (IMPORTANTE: usar fit del entrenamiento, NO fit_transform)
    X_nuevos_scaled = scaler_X_prod.transform(X_nuevos)
    
    # 3. Crear secuencias
    N_STEPS_PRED = 7
    
    def create_sequences_pred(X_data, time_steps=7):
        Xs = []
        for i in range(len(X_data) - time_steps):
            Xs.append(X_data[i : (i + time_steps)])
        return np.array(Xs)
    
    X_nuevos_seq = create_sequences_pred(X_nuevos_scaled, N_STEPS_PRED)
    
    print(f" Secuencias creadas: {X_nuevos_seq.shape}")
    print(f"   Formato: (n_muestras={X_nuevos_seq.shape[0]}, time_steps={X_nuevos_seq.shape[1]}, features={X_nuevos_seq.shape[2]})")
    
    return X_nuevos, X_nuevos_scaled, X_nuevos_seq, y_nuevos


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Hacer Predicciones
    
    Usamos el modelo cargado para predecir el stock futuro.
    """)
    return


@app.cell
def _(X_nuevos_seq, modelo_produccion, scaler_y_prod):
    # Predecir (en escala normalizada)
    predicciones_scaled = modelo_produccion.predict(X_nuevos_seq)
    
    # Invertir escala (a unidades reales)
    predicciones_reales = scaler_y_prod.inverse_transform(predicciones_scaled)
    
    print(f" Predicciones realizadas: {len(predicciones_reales)} valores")
    print(f"   Rango: {predicciones_reales.min():.2f} - {predicciones_reales.max():.2f} unidades")
    
    return predicciones_reales, predicciones_scaled


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Comparar con Valores Reales (si existen)
    
    Como estamos usando datos del mismo dataset, podemos comparar
    nuestras predicciones con los valores reales para validar.
    """)
    return


@app.cell
def _(mo, np, pd, predicciones_reales, scaler_y_prod, y_nuevos):
    # Ajustar longitud (las secuencias reducen el tamaño)
    N_STEPS_COMP = 7
    y_reales_comparacion = y_nuevos.iloc[N_STEPS_COMP:].values
    
    # Crear DataFrame de comparación
    df_comparacion = pd.DataFrame({
        'Stock Real': y_reales_comparacion.flatten(),
        'Stock Predicho': predicciones_reales.flatten()
    })
    df_comparacion['Error (unidades)'] = df_comparacion['Stock Real'] - df_comparacion['Stock Predicho']
    df_comparacion['Error (%)'] = (abs(df_comparacion['Error (unidades)']) / df_comparacion['Stock Real']) * 100
    
    # Métricas
    mae_nuevos = np.mean(np.abs(df_comparacion['Error (unidades)']))
    rmse_nuevos = np.sqrt(np.mean(df_comparacion['Error (unidades)']**2))
    
    mo.md(rf"""
    ###  Resultados de las Predicciones
    
    **Métricas en Nuevos Datos:**
    - **MAE:** {mae_nuevos:.2f} unidades
    - **RMSE:** {rmse_nuevos:.2f} unidades
    - **Error Promedio:** {df_comparacion['Error (%)'].mean():.2f}%
    
    **Primeros 10 Resultados:**
    
    | Stock Real | Stock Predicho | Error (unidades) | Error (%) |
    |------------|----------------|------------------|-----------|
    {chr(10).join([f"| {row['Stock Real']:.2f} | {row['Stock Predicho']:.2f} | {row['Error (unidades)']:+.2f} | {row['Error (%)']:.2f}% |" for _, row in df_comparacion.head(10).iterrows()])}
    """)
    
    return df_comparacion, mae_nuevos, rmse_nuevos


@app.cell
def _(df_comparacion, plt, sns):
    # Visualización
    fig_pred, axes_pred = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Serie temporal
    axes_pred[0, 0].plot(df_comparacion.index, df_comparacion['Stock Real'], 
                         label='Real', linewidth=2, alpha=0.7)
    axes_pred[0, 0].plot(df_comparacion.index, df_comparacion['Stock Predicho'], 
                         label='Predicho', linewidth=2, alpha=0.7)
    axes_pred[0, 0].set_title('Serie Temporal: Real vs Predicho')
    axes_pred[0, 0].set_xlabel('Índice')
    axes_pred[0, 0].set_ylabel('Stock (unidades)')
    axes_pred[0, 0].legend()
    axes_pred[0, 0].grid(alpha=0.3)
    
    # 2. Scatter plot
    axes_pred[0, 1].scatter(df_comparacion['Stock Real'], df_comparacion['Stock Predicho'], 
                            alpha=0.6, s=50)
    max_val = max(df_comparacion['Stock Real'].max(), df_comparacion['Stock Predicho'].max())
    min_val = min(df_comparacion['Stock Real'].min(), df_comparacion['Stock Predicho'].min())
    axes_pred[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes_pred[0, 1].set_title('Scatter: Real vs Predicho')
    axes_pred[0, 1].set_xlabel('Stock Real')
    axes_pred[0, 1].set_ylabel('Stock Predicho')
    axes_pred[0, 1].grid(alpha=0.3)
    
    # 3. Distribución de errores
    axes_pred[1, 0].hist(df_comparacion['Error (unidades)'], bins=30, edgecolor='black', alpha=0.7)
    axes_pred[1, 0].axvline(x=0, color='red', linestyle='--', lw=2)
    axes_pred[1, 0].set_title('Distribución de Errores')
    axes_pred[1, 0].set_xlabel('Error (unidades)')
    axes_pred[1, 0].set_ylabel('Frecuencia')
    axes_pred[1, 0].grid(alpha=0.3)
    
    # 4. Error porcentual
    axes_pred[1, 1].plot(df_comparacion.index, df_comparacion['Error (%)'], 
                         color='orange', linewidth=2)
    axes_pred[1, 1].axhline(y=5, color='red', linestyle='--', label='5% threshold')
    axes_pred[1, 1].set_title('Error Porcentual a lo Largo del Tiempo')
    axes_pred[1, 1].set_xlabel('Índice')
    axes_pred[1, 1].set_ylabel('Error (%)')
    axes_pred[1, 1].legend()
    axes_pred[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predicciones_nuevos_datos.png', dpi=150)
    plt.show()
    
    print(" Visualizaciones generadas")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Guardar Predicciones
    
    Exportamos las predicciones a un archivo CSV para uso futuro.
    """)
    return


@app.cell
def _(df_comparacion):
    # Guardar a CSV
    df_comparacion.to_csv('predicciones_nuevos_datos.csv', index=False)
    print(" Predicciones guardadas en 'predicciones_nuevos_datos.csv'")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()