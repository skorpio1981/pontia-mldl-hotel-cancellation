import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

from data_loader import cargar_datos, dividir_datos


# Rutas y seed para asegurar repitibilidad
PATH_PROYECTO = Path(__file__).resolve().parents[1]
PATH_DATASET = PATH_PROYECTO / "data" / "dataset_practica_final.csv"
PATH_RESULTADOS = PATH_PROYECTO / "artifacts"
RANDOM_STATE = 42
PATH_RESULTADOS.mkdir(exist_ok=True)


# Funcion para calcular las metricas independeientemente del modelo
def calcular_metricas(nombre_modelo, y_real, y_pred, y_probabilidad):
    return {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_real, y_probabilidad),
    }


# 1. Carga, limpieza y division del dataset
df_reservas, X, y, numero_duplicados = cargar_datos(PATH_DATASET)
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

# 2. Procesamiento de los datos
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

pipeline_numerico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)

pipeline_categorico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas),
    ]
)

X_train_preparado = preprocesador.fit_transform(X_train)
X_validacion_preparado = preprocesador.transform(X_validacion)
X_test_preparado = preprocesador.transform(X_test)

resultados = []
probabilidades_validacion = {}
modelos_entrenados = {}


# 3. Entrenamiento y eval de la Regresion Logistica
modelo_rl = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)
modelo_rl.fit(X_train_preparado, y_train)
y_pred_rl = modelo_rl.predict(X_validacion_preparado)
y_prob_rl = modelo_rl.predict_proba(
    X_validacion_preparado
)[:, 1]
resultados.append(
    calcular_metricas(
        "Regresion logistica",
        y_validacion,
        y_pred_rl,
        y_prob_rl,
    )
)
probabilidades_validacion["Regresion logistica"] = y_prob_rl
modelos_entrenados["Regresion logistica"] = modelo_rl

# 4. Entrenamiento y eval de Random Forest
modelo_rfc = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
modelo_rfc.fit(X_train_preparado, y_train)
y_pred_rfc = modelo_rfc.predict(X_validacion_preparado)
y_prob_rfc = modelo_rfc.predict_proba(
    X_validacion_preparado
)[:, 1]
resultados.append(
    calcular_metricas(
        "Random Forest",
        y_validacion,
        y_pred_rfc,
        y_prob_rfc,
    )
)
probabilidades_validacion["Random Forest"] = y_prob_rfc
modelos_entrenados["Random Forest"] = modelo_rfc

# 5. Comparacion de RL vs RFC en validacion, ROC-AUC es la métrica principal por la que se ordenaran
tabla_resultados = pd.DataFrame(resultados).sort_values(
    "roc_auc",
    ascending=False,
)
tabla_resultados.to_csv(
    PATH_RESULTADOS / "validation_metrics.csv",
    index=False,
)

print("\nResultados de validacion:")
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
plt.title("Comparacion ROC en validacion")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PATH_RESULTADOS / "roc_validation.png", dpi=180)
plt.close()

# 6. Eval final
nombre_mejor_modelo = tabla_resultados.iloc[0]["modelo"]
mejor_modelo = modelos_entrenados[nombre_mejor_modelo]

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

with open(PATH_RESULTADOS / "test_metrics.json", "w", encoding="utf-8") as fichero:
    json.dump(metricas_test, fichero, indent=2, ensure_ascii=False)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No cancelacion", "Cancelacion"],
    yticklabels=["No cancelacion", "Cancelacion"],
)
plt.xlabel("Prediccion")
plt.ylabel("Valor real")
plt.title(f"Matriz de confusion: {nombre_mejor_modelo}")
plt.tight_layout()
plt.savefig(PATH_RESULTADOS / "confusion_matrix_test.png", dpi=180)
plt.close()

print(f"\nModelo seleccionado: {nombre_mejor_modelo}")
print(pd.Series(metricas_test).to_string())
