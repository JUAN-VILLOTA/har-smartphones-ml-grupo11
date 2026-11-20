# Reconocimiento de actividades humanas con smartphones (UCI HAR) — Grupo 11

Clasificación supervisada multiclase de seis actividades (caminar, subir escaleras, bajar escaleras, sentado, de pie y acostado) a partir de las 561 características calculadas sobre las señales del acelerómetro y el giroscopio de un smartphone.

- **Curso:** Introducción al Aprendizaje Automático — Universidad de Antioquia, 2026-2
- **Profesor:** Julián D. Arias-Londoño
- **Integrantes:** Juan David Villota Córdoba y Sammir Alejandro Bolaños Luna
- **Base de datos:** [Human Activity Recognition Using Smartphones (UCI)](https://doi.org/10.24432/C54S4K): 30 voluntarios, 10 299 ventanas, 561 características, 6 actividades.

## Contenido del repositorio

| Archivo | Qué contiene |
| --- | --- |
| `informe_final.pdf` | Informe final (plantilla IEEE Transactions). |
| `Primer_Informe_Modelos_ll.pdf` | Primer informe de avance (descripción del problema y estado del arte). |
| `01_descripcion_y_eda.ipynb` | Descarga de los datos, revisión de calidad, balance de clases, voluntarios y figura exploratoria. |
| `02_modelos.ipynb` | Validación por voluntario, ajuste de hiperparámetros de los 5 modelos, comparación en validación y evaluación única en prueba. |
| `03_reduccion_dimension.ipynb` | Índice de Fisher y correlación, PCA y UMAP con los dos mejores modelos (LDA y MLP). |

## Cómo reproducir los resultados

1. Abra cada notebook en Google Colab con su botón **Open in Colab** (o desde *Archivo → Abrir cuaderno → GitHub* pegando la URL de este repositorio).
2. Ejecútelos **en orden** (01 → 02 → 03) con *Entorno de ejecución → Reiniciar y ejecutar todo*. Cada notebook es independiente: descarga los datos de UCI por su cuenta, así que no hace falta subir archivos.
3. El notebook 03 instala `umap-learn` en su primera celda. No se necesita GPU.

Tiempo aproximado, según las ejecuciones guardadas: el notebook 01 tarda menos de un minuto; el 02 y el 03, unos 15 minutos cada uno (las búsquedas de la SVM y de UMAP son las más lentas).

**Dependencias:** Python 3, numpy, pandas, matplotlib, scikit-learn y umap-learn.
<!-- TODO: completar con la salida de la celda de versiones del notebook 02 -->

## Metodología, en breve

- **Partición:** se respeta la partición oficial por voluntario (21 voluntarios y 7 352 ventanas de entrenamiento; 9 voluntarios y 2 947 ventanas de prueba). El identificador del voluntario **no** es una entrada del modelo.
- **Validación:** `GroupKFold` de 5 particiones por voluntario dentro del entrenamiento, iguales para todos los modelos.
- **Preprocesamiento:** el escalado, el filtro de variables, PCA y UMAP van dentro de un `Pipeline`, ajustados solo con la parte de entrenamiento de cada partición.
- **Selección:** `GridSearchCV` por F1 macro. Los resultados de validación se reportan como media ± 2 desviaciones estándar de las 5 particiones.
- **Prueba:** se usa una sola vez, con las configuraciones ya elegidas.
- **Reproducibilidad:** semilla fija (`random_state=0`) en Random Forest, MLP y UMAP; `GroupKFold` no baraja.

## Resultados principales

| Modelo | F1 macro validación (± 2 desv. est.) | F1 macro prueba | Exactitud prueba |
| --- | --- | --- | --- |
| LDA (`shrinkage` = 0,0001) | 0,9502 ± 0,0591 | 0,9633 | 0,9630 |
| MLP (1 capa de 32 neuronas) | 0,9348 ± 0,0741 | 0,9516 | 0,9511 |
| SVM RBF (C = 100, γ = 0,0001) | 0,9322 ± 0,0911 | 0,9561 | 0,9562 |
| Random Forest (200 árboles, `max_features` = 5) | 0,9223 ± 0,0869 | 0,9260 | 0,9294 |
| KNN (k = 25, Manhattan) | 0,8924 ± 0,0589 | 0,8998 | 0,9043 |

Reducción de dimensión con LDA: PCA al 95 % de varianza deja 102 componentes (81,8 % de reducción) con F1 de validación 0,9090 ± 0,0882; UMAP con 20 componentes (96,4 % de reducción) da 0,8169 ± 0,0626. El detalle está en el informe y en el notebook 03.
