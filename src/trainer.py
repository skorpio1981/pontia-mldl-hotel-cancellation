import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_loader import cargar_datos, dividir_datos


# Definición de rutas y semilla para poder repetir el experimento.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATASET = RUTA_PROYECTO / "data" / "dataset_practica_final.csv"
RUTA_RESULTADOS = RUTA_PROYECTO / "artifacts"
RANDOM_STATE = 42

RUTA_RESULTADOS.mkdir(exist_ok=True)


# ============================================================================
# FUNCIÓN AUXILIAR PARA CALCULAR LAS MISMAS MÉTRICAS EN TODOS LOS MODELOS
# Se utiliza seis veces: una por cada uno de los cinco modelos en validación y
# una última vez para evaluar el modelo ganador en test. Así evitamos copiar el
# mismo bloque de métricas seis veces y garantizamos una comparación homogénea.
# ============================================================================
def calcular_metricas(nombre_modelo, y_real, y_pred, y_probabilidad):
    """Calcula las cinco métricas utilizadas para comparar modelos."""
    return {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_real, y_probabilidad),
    }


# ============================================================================
# BLOQUE 1. CARGA, LIMPIEZA Y DIVISIÓN DE LOS DATOS
# ============================================================================
df_reservas, X, y, numero_duplicados = cargar_datos(RUTA_DATASET)
(
    X_train,
    X_validacion,
    X_test,
    y_train,
    y_validacion,
    y_test,
) = dividir_datos(X, y, RANDOM_STATE)

print(f"Filas originales: {len(df_reservas) + numero_duplicados}")
print(f"Duplicados eliminados: {numero_duplicados}")
print(f"Filas utilizadas: {len(df_reservas)}")


# ============================================================================
# BLOQUE 2. PREPROCESAMIENTO COMÚN PARA TODOS LOS MODELOS
# Se separan columnas numéricas y categóricas. El preprocesador se ajusta solo
# con train y después transforma validación y test sin aprender de ellos.
# ============================================================================
columnas_identificadores = [
    columna for columna in ["agent", "company"] if columna in X.columns
]
columnas_numericas = [
    columna
    for columna in X.select_dtypes(include=["number", "bool"]).columns
    if columna not in columnas_identificadores
]
columnas_categoricas = (
    X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    + columnas_identificadores
)

transformador_numerico = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalado", StandardScaler()),
    ]
)

transformador_categorico = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", transformador_numerico, columnas_numericas),
        ("categoricas", transformador_categorico, columnas_categoricas),
    ]
)

X_train_preparado = preprocesador.fit_transform(X_train)
X_validacion_preparado = preprocesador.transform(X_validacion)
X_test_preparado = preprocesador.transform(X_test)

resultados = []
probabilidades_validacion = {}
modelos_entrenados = {}


# ============================================================================
# BLOQUE 3. ENTRENAMIENTO Y EVALUACIÓN DE LA REGRESIÓN LOGÍSTICA
# ============================================================================
modelo_regresion_logistica = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
modelo_regresion_logistica.fit(X_train_preparado, y_train)
y_pred_regresion = modelo_regresion_logistica.predict(X_validacion_preparado)
y_prob_regresion = modelo_regresion_logistica.predict_proba(
    X_validacion_preparado
)[:, 1]
resultados.append(
    calcular_metricas(
        "Regresión logística",
        y_validacion,
        y_pred_regresion,
        y_prob_regresion,
    )
)
probabilidades_validacion["Regresión logística"] = y_prob_regresion
modelos_entrenados["Regresión logística"] = modelo_regresion_logistica


# ============================================================================
# BLOQUE 4. ENTRENAMIENTO Y EVALUACIÓN DEL ÁRBOL DE DECISIÓN
# ============================================================================
modelo_arbol = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
modelo_arbol.fit(X_train_preparado, y_train)
y_pred_arbol = modelo_arbol.predict(X_validacion_preparado)
y_prob_arbol = modelo_arbol.predict_proba(X_validacion_preparado)[:, 1]
resultados.append(
    calcular_metricas(
        "Árbol de decisión",
        y_validacion,
        y_pred_arbol,
        y_prob_arbol,
    )
)
probabilidades_validacion["Árbol de decisión"] = y_prob_arbol
modelos_entrenados["Árbol de decisión"] = modelo_arbol


# ============================================================================
# BLOQUE 5. ENTRENAMIENTO Y EVALUACIÓN DE RANDOM FOREST
# ============================================================================
modelo_random_forest = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
modelo_random_forest.fit(X_train_preparado, y_train)
y_pred_random_forest = modelo_random_forest.predict(X_validacion_preparado)
y_prob_random_forest = modelo_random_forest.predict_proba(
    X_validacion_preparado
)[:, 1]
resultados.append(
    calcular_metricas(
        "Random Forest",
        y_validacion,
        y_pred_random_forest,
        y_prob_random_forest,
    )
)
probabilidades_validacion["Random Forest"] = y_prob_random_forest
modelos_entrenados["Random Forest"] = modelo_random_forest


# ============================================================================
# BLOQUE 6. ENTRENAMIENTO Y EVALUACIÓN DE XGBOOST
# XGBoost combina árboles de forma secuencial para corregir los errores de los
# anteriores. Suele funcionar especialmente bien con datos tabulares.
# ============================================================================
modelo_xgboost = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.80,
    colsample_bytree=0.80,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
modelo_xgboost.fit(X_train_preparado, y_train)
y_pred_xgboost = modelo_xgboost.predict(X_validacion_preparado)
y_prob_xgboost = modelo_xgboost.predict_proba(X_validacion_preparado)[:, 1]
resultados.append(
    calcular_metricas("XGBoost", y_validacion, y_pred_xgboost, y_prob_xgboost)
)
probabilidades_validacion["XGBoost"] = y_prob_xgboost
modelos_entrenados["XGBoost"] = modelo_xgboost


# ============================================================================
# BLOQUE 7. ENTRENAMIENTO Y EVALUACIÓN DE LA RED NEURONAL KERAS
# La salida sigmoide devuelve una probabilidad de cancelación entre 0 y 1.
# Early Stopping detiene el entrenamiento cuando deja de mejorar el AUC.
# ============================================================================
X_train_red = (
    X_train_preparado.toarray()
    if hasattr(X_train_preparado, "toarray")
    else np.asarray(X_train_preparado)
)
X_validacion_red = (
    X_validacion_preparado.toarray()
    if hasattr(X_validacion_preparado, "toarray")
    else np.asarray(X_validacion_preparado)
)
X_test_red = (
    X_test_preparado.toarray()
    if hasattr(X_test_preparado, "toarray")
    else np.asarray(X_test_preparado)
)

tf.keras.utils.set_random_seed(RANDOM_STATE)

modelo_red_neuronal = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(X_train_red.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu", name="h1"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation="relu", name="h2"),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(1, activation="sigmoid", name="salida"),
    ]
)

modelo_red_neuronal.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

history = modelo_red_neuronal.fit(
    X_train_red,
    y_train,
    validation_split=0.15,
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=0,
)

y_prob_red = modelo_red_neuronal.predict(X_validacion_red, verbose=0).ravel()
y_pred_red = (y_prob_red >= 0.5).astype(int)
resultados.append(
    calcular_metricas(
        "Red neuronal Keras",
        y_validacion,
        y_pred_red,
        y_prob_red,
    )
)
probabilidades_validacion["Red neuronal Keras"] = y_prob_red
modelos_entrenados["Red neuronal Keras"] = modelo_red_neuronal


# ============================================================================
# BLOQUE 8. COMPARACIÓN DE LOS CINCO MODELOS EN VALIDACIÓN
# Se ordenan por ROC-AUC, que es la métrica principal elegida para la práctica.
# ============================================================================
tabla_resultados = pd.DataFrame(resultados).sort_values(
    "roc_auc",
    ascending=False,
)
tabla_resultados.to_csv(
    RUTA_RESULTADOS / "validation_metrics.csv",
    index=False,
)

print("\nResultados de validación:")
print(tabla_resultados.round(3).to_string(index=False))

plt.figure(figsize=(9, 6))
for nombre_modelo, probabilidades in probabilidades_validacion.items():
    tasa_falsos_positivos, tasa_verdaderos_positivos, _ = roc_curve(
        y_validacion,
        probabilidades,
    )
    auc_modelo = roc_auc_score(y_validacion, probabilidades)
    plt.plot(
        tasa_falsos_positivos,
        tasa_verdaderos_positivos,
        label=f"{nombre_modelo} (AUC={auc_modelo:.3f})",
    )

plt.plot([0, 1], [0, 1], "--", color="grey", label="Clasificador aleatorio")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Comparación ROC en validación")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RUTA_RESULTADOS / "roc_validation.png", dpi=180)
plt.close()


# ============================================================================
# BLOQUE 9. IMPORTANCIA DE VARIABLES DEL RANDOM FOREST
# Se utiliza Random Forest porque ofrece feature_importances_ directamente.
# ============================================================================
nombres_variables = preprocesador.get_feature_names_out()
importancia_variables = pd.Series(
    modelo_random_forest.feature_importances_,
    index=nombres_variables,
).nlargest(20).sort_values()

plt.figure(figsize=(9, 7))
importancia_variables.plot.barh(color="#2a6fbb")
plt.title("Las 20 variables más importantes (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig(RUTA_RESULTADOS / "feature_importance.png", dpi=180)
plt.close()


# ============================================================================
# BLOQUE 10. SELECCIÓN DEL GANADOR Y EVALUACIÓN FINAL EN TEST
# El test no se ha usado para entrenar ni para elegir el modelo. Aquí se utiliza
# por primera vez para obtener una estimación final imparcial.
# ============================================================================
nombre_mejor_modelo = tabla_resultados.iloc[0]["modelo"]
mejor_modelo = modelos_entrenados[nombre_mejor_modelo]

if nombre_mejor_modelo == "Red neuronal Keras":
    y_prob_test = mejor_modelo.predict(X_test_red, verbose=0).ravel()
    y_pred_test = (y_prob_test >= 0.5).astype(int)
else:
    y_prob_test = mejor_modelo.predict_proba(X_test_preparado)[:, 1]
    y_pred_test = mejor_modelo.predict(X_test_preparado)

matriz = confusion_matrix(y_test, y_pred_test)
metricas_test = calcular_metricas(
    nombre_mejor_modelo,
    y_test,
    y_pred_test,
    y_prob_test,
)
metricas_test["matriz_confusion"] = {
    "verdaderos_negativos": int(matriz[0, 0]),
    "falsos_positivos": int(matriz[0, 1]),
    "falsos_negativos": int(matriz[1, 0]),
    "verdaderos_positivos": int(matriz[1, 1]),
}

with open(RUTA_RESULTADOS / "test_metrics.json", "w", encoding="utf-8") as fichero:
    json.dump(metricas_test, fichero, indent=2, ensure_ascii=False)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No cancelación", "Cancelación"],
    yticklabels=["No cancelación", "Cancelación"],
)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title(f"Matriz de confusión: {nombre_mejor_modelo}")
plt.tight_layout()
plt.savefig(RUTA_RESULTADOS / "confusion_matrix_test.png", dpi=180)
plt.close()

print(f"\nModelo seleccionado: {nombre_mejor_modelo}")
print(pd.Series(metricas_test).to_string())


# ============================================================================
# BLOQUE 11. GUARDADO DEL MODELO Y EJEMPLO DE PREDICCIÓN
# Se guarda conjuntamente el preprocesador y el modelo para poder transformar
# reservas futuras exactamente de la misma forma que los datos de entrenamiento.
# ============================================================================
if nombre_mejor_modelo != "Red neuronal Keras":
    joblib.dump(
        {"preprocesador": preprocesador, "modelo": mejor_modelo},
        RUTA_RESULTADOS / "best_model.joblib",
    )

print("\nEjemplo de predicciones sobre cinco reservas de test:")
print(
    pd.DataFrame(
        {
            "probabilidad_cancelacion": y_prob_test[:5],
            "prediccion": y_pred_test[:5],
            "valor_real": y_test.iloc[:5].values,
        }
    ).round(3)
)
