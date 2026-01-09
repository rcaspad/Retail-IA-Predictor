import streamlit as st
import pickle
import pandas as pd
from datetime import datetime, timedelta
from prophet.plot import plot_plotly

# Configuración de la página
st.set_page_config(page_title="Predicción de Ventas", layout="wide")

# Título
st.markdown("# 📈 Predicción de Ventas Futuras")

st.markdown("""
Utiliza nuestro modelo Prophet entrenado para generar predicciones de ventas futuras.
Selecciona el período de predicción y genera pronósticos precisos.
""")

# Separador
st.divider()

# Configuración de predicción
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown("### ⚙️ Configuración")
    prediction_days = st.select_slider(
        "Días a predecir",
        options=[30, 60, 90],
        value=30
    )

with col2:
    st.markdown("### ")
    st.markdown("&nbsp;")
    generate_button = st.button("🔮 Generar Predicción", use_container_width=True, type="primary")

with col3:
    st.markdown("### 📌 Información")
    st.info(f"Se generará una predicción de **{prediction_days} días** a partir de hoy.")

st.divider()

# Cargar modelo y generar predicción
if generate_button:
    try:
        # Intentar cargar el modelo
        with st.spinner("📂 Cargando modelo de Prophet..."):
            try:
                with open("models/sales_model.pkl", "rb") as f:
                    model = pickle.load(f)
                st.success("✅ Modelo cargado correctamente")
            except FileNotFoundError:
                st.error("❌ Archivo de modelo no encontrado en 'models/sales_model.pkl'")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error al cargar el modelo: {str(e)}")
                st.stop()

        # Generar predicción futura
        with st.spinner(f"🔄 Generando predicción para {prediction_days} días..."):
            try:
                # Crear dataframe futuro
                future = model.make_future_dataframe(periods=prediction_days)
                
                # Hacer predicción
                forecast = model.predict(future)
                
                st.success("✅ Predicción generada exitosamente")
                
                # Mostrar gráfico interactivo
                st.markdown("### 📊 Gráfico de Predicción")
                fig = plot_plotly(model, forecast)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar tabla con últimos 5 días predichos
                st.markdown("### 📋 Últimos 5 Días Predichos")
                
                # Filtrar solo datos futuros
                forecast_only = forecast[forecast['ds'] > forecast['ds'].max() - timedelta(days=prediction_days)]
                
                # Seleccionar columnas relevantes y últimas 5 filas
                display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                last_5_forecast = forecast_only[display_cols].tail(5).copy()
                
                # Renombrar columnas para mejor presentación
                last_5_forecast.columns = ['Fecha', 'Predicción', 'Límite Inferior', 'Límite Superior']
                last_5_forecast['Fecha'] = last_5_forecast['Fecha'].dt.strftime('%Y-%m-%d')
                
                # Formatear números a 2 decimales
                for col in ['Predicción', 'Límite Inferior', 'Límite Superior']:
                    last_5_forecast[col] = last_5_forecast[col].round(2)
                
                st.dataframe(
                    last_5_forecast,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Predicción": st.column_config.NumberColumn(format="$%.2f"),
                        "Límite Inferior": st.column_config.NumberColumn(format="$%.2f"),
                        "Límite Superior": st.column_config.NumberColumn(format="$%.2f")
                    }
                )
                
                # Mostrar estadísticas resumen
                st.markdown("### 📊 Resumen Estadístico")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_prediction = forecast_only['yhat'].mean()
                    st.metric("Promedio Predicho", f"${avg_prediction:.2f}")
                
                with col2:
                    max_prediction = forecast_only['yhat'].max()
                    st.metric("Máximo Predicho", f"${max_prediction:.2f}")
                
                with col3:
                    min_prediction = forecast_only['yhat'].min()
                    st.metric("Mínimo Predicho", f"${min_prediction:.2f}")
                
                with col4:
                    total_prediction = forecast_only['yhat'].sum()
                    st.metric("Total Predicho", f"${total_prediction:.2f}")
                
            except Exception as e:
                st.error(f"❌ Error al generar la predicción: {str(e)}")
                st.markdown("""
                ### 🔧 Posibles soluciones:
                - Verifica que el modelo esté correctamente entrenado
                - Asegúrate que los datos de entrada sean válidos
                - Intenta con un período diferente
                """)
    
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")

# Información de ayuda
st.markdown("""
---
### 💡 Cómo usar esta página:

1. **Selecciona el período**: Elige entre 30, 60 o 90 días de predicción
2. **Genera la predicción**: Haz clic en el botón "Generar Predicción"
3. **Analiza los resultados**: 
   - El gráfico muestra la tendencia histórica y la predicción futura
   - La tabla detalla los últimos 5 días predichos
   - Las estadísticas resumen dan una visión general

### 📌 Notas importantes:

- Los intervalos de confianza (líneas punteadas) representan la incertidumbre de la predicción
- Cuanto más lejano es el futuro, mayor es la incertidumbre
- Los datos se actualizan automáticamente con nueva información disponible
""")
