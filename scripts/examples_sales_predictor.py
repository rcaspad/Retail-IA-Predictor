"""
Ejemplos de uso del SalesTimeSeriesPredictor

Este script demuestra diferentes formas de usar la clase
para predicción de ventas con Prophet.
"""

from src.models.sales_predictor import SalesTimeSeriesPredictor
import pandas as pd


def ejemplo_1_entrenamiento_basico():
    """Ejemplo 1: Entrenamiento básico del modelo"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Entrenamiento Básico")
    print("="*70)
    
    # Crear y entrenar predictor
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    # Realizar predicción simple
    forecast = predictor.predict_next_days(days=7)
    
    print("\nPredicciones para los próximos 7 días:")
    print(forecast.to_string())


def ejemplo_2_prediccion_multiple():
    """Ejemplo 2: Predicciones para diferentes períodos"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Predicciones Múltiples")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    # Predicciones para diferentes períodos
    periodos = [7, 14, 30, 90]
    
    for dias in periodos:
        forecast = predictor.predict_next_days(days=dias)
        promedio = forecast['yhat'].mean()
        print(f"Promedio de ventas predichas (próximos {dias} días): ${promedio:.2f}")


def ejemplo_3_analisis_estadistico():
    """Ejemplo 3: Análisis estadístico de predicciones"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Análisis Estadístico")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    forecast = predictor.predict_next_days(days=30)
    
    print("\nEstadísticas de predicción (próximos 30 días):")
    print(f"Mínimo: ${forecast['yhat'].min():.2f}")
    print(f"Máximo: ${forecast['yhat'].max():.2f}")
    print(f"Promedio: ${forecast['yhat'].mean():.2f}")
    print(f"Desv. Estándar: ${forecast['yhat'].std():.2f}")
    print(f"Mediana: ${forecast['yhat'].median():.2f}")
    
    # Analizar amplitud de incertidumbre
    forecast['incertidumbre'] = forecast['yhat_upper'] - forecast['yhat_lower']
    print(f"\nIncertidumbre promedio (rango de confianza): ${forecast['incertidumbre'].mean():.2f}")


def ejemplo_4_comparacion_periodos():
    """Ejemplo 4: Comparación entre períodos"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Comparación de Períodos")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    # Predicciones por semana
    forecast_30 = predictor.predict_next_days(days=30)
    
    # Dividir en semanas
    forecast_30['semana'] = (forecast_30['ds'].dt.isocalendar().week)
    
    print("\nVentas predichas por semana:")
    weekly_avg = forecast_30.groupby('semana')['yhat'].agg(['mean', 'min', 'max'])
    print(weekly_avg.to_string())


def ejemplo_5_exportar_resultados():
    """Ejemplo 5: Exportar predicciones a CSV"""
    print("\n" + "="*70)
    print("EJEMPLO 5: Exportar Resultados")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    forecast = predictor.predict_next_days(days=30)
    
    # Guardar a CSV
    archivo_salida = 'data/processed/forecast_30days.csv'
    forecast.to_csv(archivo_salida, index=False)
    print(f"\n✅ Predicciones guardadas en: {archivo_salida}")
    
    # Mostrar primeras filas
    print("\nPrimeras 5 filas del archivo guardado:")
    df_guardado = pd.read_csv(archivo_salida)
    print(df_guardado.head().to_string())


def ejemplo_6_prediccion_mañana():
    """Ejemplo 6: Enfoque especializado en predicción de mañana"""
    print("\n" + "="*70)
    print("EJEMPLO 6: Predicción para Mañana")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    # Obtener predicción para mañana
    tomorrow = predictor.get_tomorrow_prediction()
    
    if tomorrow:
        print(f"\n📅 Fecha: {tomorrow['date']}")
        print(f"💰 Predicción: ${tomorrow['yhat']:.2f}")
        print(f"📊 Intervalo de confianza (95%):")
        print(f"   Mínimo: ${tomorrow['yhat_lower']:.2f}")
        print(f"   Máximo: ${tomorrow['yhat_upper']:.2f}")
        
        # Calcular margen de incertidumbre
        margen = (tomorrow['yhat_upper'] - tomorrow['yhat_lower']) / 2
        porcentaje = (margen / tomorrow['yhat']) * 100
        print(f"   Margen de error: ±${margen:.2f} ({porcentaje:.1f}%)")


def ejemplo_7_cargar_modelo_preentrenado():
    """Ejemplo 7: Usar un modelo preentrenado"""
    print("\n" + "="*70)
    print("EJEMPLO 7: Cargar Modelo Preentrenado")
    print("="*70)
    
    # Crear predictor y cargar modelo existente
    predictor = SalesTimeSeriesPredictor()
    
    try:
        predictor.load_model()
        
        # Usar el modelo cargado para predicción
        forecast = predictor.predict_next_days(days=7)
        
        print("\n✅ Modelo cargado correctamente")
        print(f"Predicción para los próximos 7 días (modelo preentrenado):")
        print(forecast[['ds', 'yhat']].to_string())
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Primero debes ejecutar: predictor.train()")


def ejemplo_8_analisis_tendencia():
    """Ejemplo 8: Análisis de tendencia"""
    print("\n" + "="*70)
    print("EJEMPLO 8: Análisis de Tendencia")
    print("="*70)
    
    predictor = SalesTimeSeriesPredictor()
    predictor.train()
    
    # Predicciones para 90 días
    forecast = predictor.predict_next_days(days=90)
    
    # Dividir en trimestres
    print("\nAnálisis por trimestre (90 días):")
    print(f"Primera semana (días 1-7):")
    print(f"  Promedio: ${forecast.iloc[:7]['yhat'].mean():.2f}")
    
    print(f"\nSegunda-cuarta semana (días 8-30):")
    print(f"  Promedio: ${forecast.iloc[7:30]['yhat'].mean():.2f}")
    
    print(f"\nResto (días 31-90):")
    print(f"  Promedio: ${forecast.iloc[30:]['yhat'].mean():.2f}")
    
    # Calcular tendencia
    primera_semana = forecast.iloc[:7]['yhat'].mean()
    ultima_semana = forecast.iloc[-7:]['yhat'].mean()
    cambio_porcentual = ((ultima_semana - primera_semana) / primera_semana) * 100
    
    print(f"\nTendencia general:")
    print(f"  Cambio: {cambio_porcentual:+.2f}%")
    if cambio_porcentual > 0:
        print(f"  📈 Tendencia alcista detectada")
    elif cambio_porcentual < 0:
        print(f"  📉 Tendencia bajista detectada")
    else:
        print(f"  ➡️ Tendencia estable")


def main():
    """Ejecutar todos los ejemplos"""
    print("\n" + "🚀 "*35)
    print("EJEMPLOS DE USO - SALES TIME SERIES PREDICTOR")
    print("🚀 "*35)
    
    try:
        # Descomentar los ejemplos que deseas ejecutar
        ejemplo_1_entrenamiento_basico()
        # ejemplo_2_prediccion_multiple()
        # ejemplo_3_analisis_estadistico()
        # ejemplo_4_comparacion_periodos()
        # ejemplo_5_exportar_resultados()
        # ejemplo_6_prediccion_mañana()
        # ejemplo_7_cargar_modelo_preentrenado()
        # ejemplo_8_analisis_tendencia()
        
        print("\n" + "="*70)
        print("✅ Ejemplos completados exitosamente")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
