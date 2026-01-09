import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Retail IA", layout="wide")

# Título principal
st.markdown("# 🛒 Retail IA - Sistema Predictivo")

# Descripción profesional
st.markdown("""
### 📊 Bienvenido al Sistema de Predicción Inteligente

Nuestro sistema utiliza tecnologías de **Machine Learning avanzadas** para optimizar tu negocio:

- **Prophet**: Modelo estadístico robusto para capturar tendencias y patrones estacionales en series temporales
- **XGBoost**: Algoritmo de gradient boosting para predicciones precisas y generalizables

Este sistema te permite:
- ✅ Predecir ventas futuras con alta precisión
- ✅ Identificar patrones de comportamiento de clientes
- ✅ Optimizar la gestión de inventario
- ✅ Maximizar márgenes de ganancia
- ✅ Tomar decisiones estratégicas basadas en datos

---
""")

# Sección de KPIs
st.markdown("### 📈 Métricas Clave del Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💰 Ventas Totales",
        value="$2,547,890",
        delta="+12.5%",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="👥 Clientes Activos",
        value="3,842",
        delta="+8.3%",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="📊 Margen Promedio",
        value="34.2%",
        delta="+2.1%",
        delta_color="normal"
    )

# Información adicional
st.markdown("""
---
### 🚀 Comenzar

Selecciona una opción en el menú lateral para:
- **Predicción de Ventas**: Genera predicciones de ventas futuras
- **Análisis de Clientes**: Identifica patrones de comportamiento

El sistema se actualiza automáticamente con nuevos datos y modelos entrenados.
""")

# Footer
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: 12px;">
Retail IA © 2026 | Sistema de Predicción Inteligente
</div>
""", unsafe_allow_html=True)
