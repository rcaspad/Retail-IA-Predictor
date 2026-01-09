# 🛒 Retail IA Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Machine Learning](https://img.shields.io/badge/Models-Prophet%20%7C%20XGBoost-green)
![Status](https://img.shields.io/badge/Status-Active-success)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-ia-predictor-bgy835kdtgkhqrakrnszkc.streamlit.app/)

**Sistema End-to-End de Inteligencia Artificial para Retail.** Esta solución permite predecir ventas futuras, identificar clientes en riesgo de abandono (Churn) y visualizar métricas clave de negocio mediante un dashboard interactivo.

---

## 🚀 Características Principales

### 1. 📈 Predicción de Ventas (Sales Forecasting)
- **Motor:** Facebook Prophet.
- **Funcionalidad:** Predice el volumen de ventas diario con intervalos de confianza.
- **Capacidades:** Detecta estacionalidad (ej: picos en verano) y tendencias de crecimiento a largo plazo.

### 2. 🔄 Detección de Fugas (Churn Prediction)
- **Motor:** XGBoost Classifier.
- **Funcionalidad:** Calcula la probabilidad de que un cliente deje de comprar en los próximos 90 días.
- **Insights:** Analiza Recency (tiempo desde última compra), Frecuencia y Ticket Promedio.

### 3. 💻 Dashboard Interactivo
- Interfaz web construida con **Streamlit**.
- Gráficos dinámicos con **Plotly**.
- Generación de reportes y listas de clientes en riesgo exportables a CSV.

---

## 📂 Estructura del Proyecto

El proyecto sigue estándares de Data Science (Cookiecutter) para máxima escalabilidad:

```text
Retail-IA-Predictor/
├── app/                # Código de la aplicación Streamlit (Frontend)
│   ├── Home.py         # Página de inicio
│   └── pages/          # Módulos de Ventas y Churn
├── data/               # Almacenamiento de datos (Raw y Processed)
├── docs/               # Documentación y guías del proyecto
├── models/             # Modelos entrenados (.pkl)
├── scripts/            # Scripts de demostración y ejemplos
├── src/                # Código fuente núcleo (ETL, Entrenamiento)
│   ├── data/           # Scripts de generación y limpieza de datos
│   └── models/         # Lógica de entrenamiento de modelos
├── .gitignore          # Configuración de Git
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación principal
