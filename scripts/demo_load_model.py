"""
Script de demostración - Cargar modelo y realizar predicción
"""

from src.models.sales_predictor import SalesTimeSeriesPredictor

print("\n" + "="*70)
print("🎯 DEMOSTRACIÓN: Cargar Modelo y Predicción para Mañana")
print("="*70 + "\n")

try:
    # Crear instancia
    predictor = SalesTimeSeriesPredictor()
    
    # Cargar modelo preentrenado
    print("[1] Cargando modelo preentrenado...")
    predictor.load_model()
    
    # Obtener predicción para mañana
    print("[2] Obteniendo predicción para mañana...")
    tomorrow = predictor.get_tomorrow_prediction()
    
    # Mostrar resultados
    if tomorrow:
        print("\n" + "-"*70)
        print("✅ PREDICCIÓN EXITOSA")
        print("-"*70)
        print(f"📅 Fecha: {tomorrow['date']}")
        print(f"💰 Ventas predichas: ${tomorrow['yhat']:.2f}")
        print(f"📊 Intervalo de confianza (95%):")
        print(f"   Mínimo: ${tomorrow['yhat_lower']:.2f}")
        print(f"   Máximo: ${tomorrow['yhat_upper']:.2f}")
        
        # Calcular margen
        margen = (tomorrow['yhat_upper'] - tomorrow['yhat_lower']) / 2
        porcentaje = (margen / tomorrow['yhat']) * 100
        print(f"   Margen de error: ±${margen:.2f} ({porcentaje:.1f}%)")
        print("-"*70 + "\n")
    
    # Predicción adicional para 7 días
    print("[3] Generando predicción para los próximos 7 días...")
    forecast_7 = predictor.predict_next_days(days=7)
    
    print("\n📈 Predicciones de 7 días:")
    for idx, row in forecast_7.iterrows():
        print(f"   {row['ds'].date()}: ${row['yhat']:.2f}")
    
    print("\n" + "="*70)
    print("✨ Demostración completada exitosamente")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n❌ Error: {e}\n")
