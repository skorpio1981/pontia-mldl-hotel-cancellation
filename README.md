# Predicción de cancelaciones hoteleras

Práctica final de Machine Learning y Deep Learning de PontIA. El objetivo es
predecir si una reserva de hotel será cancelada (`is_canceled = 1`).

El proyecto compara los cinco algoritmos solicitados: regresión logística,
árbol de decisión, Random Forest, XGBoost y red neuronal con Keras.

## Qué contiene cada carpeta

```text
data/
  dataset_practica_final.csv       datos proporcionados por PontIA
notebooks/
  01_analisis_exploratorio.ipynb  conocimiento y limpieza inicial de los datos
  02_entrenamiento_modelos.ipynb  entrenamiento explicado paso a paso
src/
  data_loader.py                   carga, limpieza y división train/validación/test
  trainer.py                       ejecuta todo el entrenamiento automáticamente
artifacts/                         métricas y gráficas generadas
docs/
  informe_final.md                 borrador del informe de entrega
  guia_defensa.md                  guion y preguntas para la defensa
```

Los notebooks son la parte principal para estudiar y presentar. Los dos
archivos de `src/` existen porque el enunciado también pide código modular y
un flujo automatizado. No hay clases personalizadas, argumentos de consola ni
una arquitectura compleja: los modelos aparecen escritos uno por uno, igual
que en un notebook de clase.

## Equipo y reparto propuesto

| Integrante | Responsabilidad principal |
| --- | --- |
| **Gabriel García Vázquez** | Integración, carga/división de datos y ejecución final |
| **Daniel Ruíz** | Análisis exploratorio y explicación de los datos |
| **Valeria [apellidos]** | Comparación de modelos, métricas e informe |

Es una propuesta. Antes de entregar, sustituid los corchetes y describid el
trabajo real de cada persona. Los commits no deben repartirse artificialmente.

## Cómo empezar desde cero

1. El CSV `data/dataset_practica_final.csv` forma parte del repositorio, por lo
   que `git clone` ya lo descarga junto con el código.
2. Abre una terminal en la carpeta raíz del repositorio.
3. Crea y activa el entorno. Hay dos formas equivalentes, usa la que prefieras:

   Con `pip`:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Con `uv` (más rápido, usa `uv.lock` para fijar versiones exactas):
   ```bash
   uv sync
   source .venv/bin/activate
   ```

4. Abre `notebooks/01_analisis_exploratorio.ipynb` en VS Code, Jupyter o
   Google Colab, léelo y ejecuta sus celdas de arriba abajo.
5. Haz lo mismo con `notebooks/02_entrenamiento_modelos.ipynb`.
6. Para comprobar el flujo completo con una sola orden, ejecuta:

   ```bash
   python src/trainer.py
   ```

Las gráficas y métricas quedarán en `artifacts/`.

## Decisiones importantes

- Se eliminan 31.994 duplicados exactos antes de dividir los datos. Quedan
  87.396 reservas únicas.
- Se excluyen `reservation_status` y `reservation_status_date` porque revelan
  información conocida después del desenlace. Usarlas sería fuga de datos.
- La división es estratificada: 70 % entrenamiento, 15 % validación y 15 %
  test. Los cinco modelos se comparan en validación y el ganador se evalúa una
  sola vez en test.
- El preprocesamiento se aprende únicamente con entrenamiento: mediana y
  escalado para números; moda y One-Hot Encoding para categorías.
- La métrica principal es ROC-AUC porque hay desbalance (27,49 % de
  cancelaciones) y permite comparar el poder discriminante sin depender de un
  único umbral. También se muestran accuracy, precision, recall y F1.

## Resultados obtenidos

| Modelo (validación) | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0,810 | 0,696 | 0,546 | 0,612 | 0,869 |
| Árbol de decisión | 0,779 | 0,565 | **0,854** | 0,680 | 0,881 |
| Random Forest | 0,782 | 0,570 | 0,844 | 0,681 | 0,889 |
| **XGBoost** | 0,846 | **0,762** | 0,638 | 0,695 | **0,915** |
| Red neuronal Keras | **0,848** | 0,723 | 0,724 | **0,724** | 0,912 |

XGBoost es el modelo seleccionado por ROC-AUC. En test obtiene accuracy 0,843,
precision 0,756, recall 0,636, F1 0,691 y ROC-AUC 0,909.

## Limitaciones

La división es aleatoria; en una aplicación real sería preferible validar por
fechas. También habría que confirmar qué variables existen en el momento exacto
de la predicción y elegir el umbral según el coste de falsos positivos y falsos
negativos.
