"""
Demo: Cargar modelo de Churn y hacer predicciones
"""

from src.models.churn_predictor import ChurnPredictor
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("🔴 DEMOSTRACIÓN: Churn Predictor - Predicción de Abandono")
print("="*70 + "\n")

try:
    # 1. Cargar modelo preentrenado
    print("[1] Cargando modelo preentrenado...")
    predictor = ChurnPredictor()
    predictor.load_model()
    
    # 2. Crear clientes de ejemplo
    print("[2] Creando clientes de ejemplo...")
    customers_example = pd.DataFrame({
        'customer_id': [101, 102, 103, 104, 105],
        'frequency': [2, 15, 8, 30, 1],
        'monetary': [1000, 8000, 4000, 15000, 300],
        'avg_ticket': [500, 533, 500, 500, 300]
    })
    
    # 3. Hacer predicciones
    print("[3] Realizando predicciones...")
    predictions = predictor.predict(customers_example[['frequency', 'monetary', 'avg_ticket']])
    probabilities = predictor.predict_churn_probability(customers_example[['frequency', 'monetary', 'avg_ticket']])
    
    customers_example['churn_prediction'] = predictions
    customers_example['churn_probability'] = probabilities
    
    # 4. Mostrar resultados
    print("\n" + "-"*70)
    print("📊 RESULTADOS DE PREDICCIÓN")
    print("-"*70)
    
    for idx, row in customers_example.iterrows():
        pred = "🔴 EN RIESGO" if row['churn_prediction'] == 1 else "🟢 ACTIVO"
        risk_level = "ALTO" if row['churn_probability'] > 0.7 else "MEDIO" if row['churn_probability'] > 0.4 else "BAJO"
        print(f"\nCliente {row['customer_id']}:")
        print(f"  Compras: {row['frequency']}, Gasto: ${row['monetary']:.0f}, Ticket Promedio: ${row['avg_ticket']:.0f}")
        print(f"  Predicción: {pred}")
        print(f"  Riesgo de Churn: {row['churn_probability']*100:.1f}% ({risk_level})")
    
    # 5. Feature Importance
    print("\n" + "-"*70)
    print("🔍 IMPORTANCIA DE FEATURES")
    print("-"*70)
    importance = predictor.get_feature_importance()
    for idx, row in importance.iterrows():
        bar_length = int(row['importance'] * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {row['feature']:15} | {bar} {row['importance']*100:.2f}%")
    
    print("\n💡 INSIGHT:")
    print("   La frecuencia de compra (59.51%) es la métrica más importante")
    print("   para predecir abandono de clientes. Clientes que compran")
    print("   con baja frecuencia tienen mucho mayor riesgo.")
    
    # 6. Recomendaciones
    print("\n" + "-"*70)
    print("💼 RECOMENDACIONES OPERACIONALES")
    print("-"*70)
    
    high_risk = customers_example[customers_example['churn_probability'] > 0.7]
    print(f"\n🚨 Clientes en ALTO RIESGO ({len(high_risk)}):")
    for _, c in high_risk.iterrows():
        print(f"   • Cliente {c['customer_id']}: {c['churn_probability']*100:.0f}% riesgo")
        print(f"     Acción: Contacto personalizado + Oferta especial")
    
    medium_risk = customers_example[(customers_example['churn_probability'] >= 0.4) & 
                                   (customers_example['churn_probability'] < 0.7)]
    if len(medium_risk) > 0:
        print(f"\n⚠️  Clientes en RIESGO MEDIO ({len(medium_risk)}):")
        for _, c in medium_risk.iterrows():
            print(f"   • Cliente {c['customer_id']}: {c['churn_probability']*100:.0f}% riesgo")
            print(f"     Acción: Monitoreo, ofertas dinámicas")
    
    low_risk = customers_example[customers_example['churn_probability'] < 0.4]
    if len(low_risk) > 0:
        print(f"\n✅ Clientes en BAJO RIESGO ({len(low_risk)}):")
        for _, c in low_risk.iterrows():
            print(f"   • Cliente {c['customer_id']}: {c['churn_probability']*100:.0f}% riesgo")
            print(f"     Acción: Comunicaciones regulares")
    
    print("\n" + "="*70)
    print("✨ Demo completada exitosamente")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n❌ Error: {e}\n")
