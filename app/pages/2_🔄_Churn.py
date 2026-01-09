import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Detector de Churn", layout="wide")

st.markdown("# 🚨 Detector de Fugas de Clientes")
st.markdown("""
Identifica rápidamente qué clientes tienen mayor probabilidad de abandono y prioriza acciones de retención.
""")

DATA_PATH = Path("data/processed/customer_features.csv")
MODEL_PATH = Path("models/churn_model.pkl")
REQUIRED_FEATURES = ["frequency", "monetary", "avg_ticket"]


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# Carga de recursos
with st.spinner("📂 Cargando datos y modelo..."):
    if not DATA_PATH.exists():
        st.error("❌ No se encontró el dataset en data/processed/customer_features.csv")
        st.stop()
    if not MODEL_PATH.exists():
        st.error("❌ No se encontró el modelo en models/churn_model.pkl")
        st.stop()

    try:
        data = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"❌ Error al leer el dataset: {exc}")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"❌ Error al cargar el modelo: {exc}")
        st.stop()

# Validación de columnas requeridas
missing_cols = {"customer_id", "recency", "frequency", "monetary"} - set(data.columns)
if missing_cols:
    st.error(f"❌ Faltan columnas en el dataset: {', '.join(sorted(missing_cols))}")
    st.stop()

# Recalcular avg_ticket como en el entrenamiento
frequency_safe = data["frequency"].replace(0, pd.NA)
data["avg_ticket"] = (data["monetary"] / frequency_safe).fillna(0)

# Preparar matriz de características
features = data[REQUIRED_FEATURES].copy()

# Predicción de probabilidad de churn
try:
    churn_proba = model.predict_proba(features)[:, 1]
except Exception as exc:
    st.error(f"❌ Error al generar predicciones: {exc}")
    st.stop()

# Anexar probabilidad al dataset original
data["Churn_Probability"] = churn_proba

st.divider()

# Filtros
col_threshold, col_info = st.columns([1, 2])
with col_threshold:
    threshold_pct = st.slider(
        "Umbral de Riesgo",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
        help="Clientes con probabilidad superior al umbral serán marcados como en riesgo."
    )
    threshold = threshold_pct / 100
with col_info:
    st.info(
        "El umbral se expresa en porcentaje. Ejemplo: 70% filtrará clientes con probabilidad > 0.70.",
        icon="ℹ️",
    )

# Filtrar clientes en riesgo
risk_df = data[data["Churn_Probability"] > threshold].copy()
risk_df.sort_values(by="Churn_Probability", ascending=False, inplace=True)

# Métricas principales
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total Clientes en Riesgo", f"{len(risk_df):,}")
with col_b:
    money_risk = risk_df["monetary"].sum()
    st.metric("Dinero en Riesgo (Suma Monetary)", f"${money_risk:,.2f}")
with col_c:
    avg_prob = risk_df["Churn_Probability"].mean() if not risk_df.empty else 0
    st.metric("Probabilidad Promedio", f"{avg_prob:.1%}")

st.divider()

# Tabla de clientes en riesgo
st.markdown("### 📋 Clientes en Riesgo")

if risk_df.empty:
    st.warning("No hay clientes por encima del umbral seleccionado.")
else:
    table = risk_df[["customer_id", "recency", "monetary", "Churn_Probability"]].rename(
        columns={
            "customer_id": "Customer ID",
            "recency": "Recency",
            "monetary": "Monetary",
            "Churn_Probability": "Churn Probability",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Customer ID": st.column_config.NumberColumn(format="%d"),
            "Recency": st.column_config.NumberColumn(format="%d"),
            "Monetary": st.column_config.NumberColumn(format="$%.2f"),
            "Churn Probability": st.column_config.ProgressColumn(
                "Prob. Churn",
                format="{:.1%}",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar clientes en riesgo",
        data=csv_bytes,
        file_name="clientes_en_riesgo.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

st.caption(
    "La probabilidad se calcula con el modelo XGBoost entrenado sobre las características frequency, monetary y avg_ticket recalculada."
)
