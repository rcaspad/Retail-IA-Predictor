"""
Ejemplos de uso del ChurnPredictor

Este script demuestra diferentes formas de usar la clase
para predicción de abandono de clientes.
"""

from src.models.churn_predictor import ChurnPredictor
import pandas as pd
import numpy as np


def ejemplo_1_entrenamiento_basico():
    """Ejemplo 1: Entrenamiento básico"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Entrenamiento Básico del Modelo")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train(test_size=0.2)
    
    print(f"\n✅ Modelo entrenado exitosamente")
    print(f"   Accuracy: {predictor.metrics['test_accuracy']*100:.2f}%")


def ejemplo_2_cargar_modelo():
    """Ejemplo 2: Cargar modelo preentrenado"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Cargar Modelo Preentrenado")
    print("="*70)
    
    predictor = ChurnPredictor()
    
    try:
        predictor.load_model()
        print("✅ Modelo cargado correctamente desde pickle")
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Ejecuta primero: predictor.train()")


def ejemplo_3_importancia_features():
    """Ejemplo 3: Analizar importancia de features"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Importancia de Features")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    importance = predictor.get_feature_importance()
    
    print("\nFeatures ordenadas por importancia:")
    print(importance.to_string())
    
    print("\nInterpretación:")
    print(f"1. '{importance.iloc[0]['feature']}' ({importance.iloc[0]['importance']*100:.1f}%)")
    print(f"   → La característica más importante para predecir churn")
    print(f"2. '{importance.iloc[1]['feature']}' ({importance.iloc[1]['importance']*100:.1f}%)")
    print(f"3. '{importance.iloc[2]['feature']}' ({importance.iloc[2]['importance']*100:.1f}%)")


def ejemplo_4_prediccion_probabilidad():
    """Ejemplo 4: Predicciones probabilísticas"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Predicción de Probabilidad de Churn")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    # Crear datos de ejemplo
    new_customers = pd.DataFrame({
        'frequency': [1, 5, 10, 20],
        'monetary': [500, 2500, 5000, 10000],
        'avg_ticket': [500, 500, 500, 500]
    })
    
    probabilities = predictor.predict_churn_probability(new_customers)
    
    print("\nClientes ejemplo con probabilidad de churn:")
    for idx, (prob, row) in enumerate(zip(probabilities, new_customers.iterrows())):
        freq = row[1]['frequency']
        monetary = row[1]['monetary']
        risk = "🔴 ALTO" if prob > 0.7 else "🟡 MEDIO" if prob > 0.4 else "🟢 BAJO"
        print(f"Cliente {idx+1}: Freq={freq}, Gasto=${monetary:.0f} → {prob*100:.1f}% riesgo {risk}")


def ejemplo_5_clasificacion_binaria():
    """Ejemplo 5: Clasificación binaria (0/1)"""
    print("\n" + "="*70)
    print("EJEMPLO 5: Clasificación Binaria")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    # Datos de ejemplo
    new_customers = pd.DataFrame({
        'frequency': [2, 15, 8, 30, 1],
        'monetary': [1000, 8000, 4000, 15000, 300],
        'avg_ticket': [500, 533, 500, 500, 300]
    })
    
    predictions = predictor.predict(new_customers)
    
    print("\nPredicciones (0=Activo, 1=Churn):")
    for idx, pred in enumerate(predictions):
        status = "⚠️  EN RIESGO" if pred == 1 else "✓ ACTIVO"
        print(f"Cliente {idx+1}: {pred} {status}")


def ejemplo_6_segmentacion_clientes():
    """Ejemplo 6: Segmentar clientes por riesgo"""
    print("\n" + "="*70)
    print("EJEMPLO 6: Segmentación de Clientes por Riesgo")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    # Crear datos de ejemplo con más clientes
    np.random.seed(42)
    n_clientes = 100
    customers = pd.DataFrame({
        'customer_id': range(1, n_clientes+1),
        'frequency': np.random.poisson(8, n_clientes),
        'monetary': np.random.normal(3000, 1500, n_clientes),
        'avg_ticket': np.random.normal(500, 200, n_clientes)
    })
    
    # Asegurar valores positivos
    customers['frequency'] = customers['frequency'].clip(lower=1)
    customers['monetary'] = customers['monetary'].clip(lower=100)
    customers['avg_ticket'] = customers['avg_ticket'].clip(lower=50)
    
    # Predecir riesgo
    probs = predictor.predict_churn_probability(customers[['frequency', 'monetary', 'avg_ticket']])
    customers['churn_probability'] = probs
    
    # Segmentar
    high_risk = customers[customers['churn_probability'] > 0.7]
    medium_risk = customers[(customers['churn_probability'] >= 0.4) & 
                            (customers['churn_probability'] <= 0.7)]
    low_risk = customers[customers['churn_probability'] < 0.4]
    
    print(f"\n📊 Segmentación de {n_clientes} clientes:")
    print(f"🔴 ALTO RIESGO (>70%):    {len(high_risk):3} clientes ({len(high_risk)/n_clientes*100:.1f}%)")
    print(f"🟡 RIESGO MEDIO (40-70%): {len(medium_risk):3} clientes ({len(medium_risk)/n_clientes*100:.1f}%)")
    print(f"🟢 BAJO RIESGO (<40%):    {len(low_risk):3} clientes ({len(low_risk)/n_clientes*100:.1f}%)")
    
    print("\nTop 5 clientes con mayor riesgo:")
    top_risk = customers.nlargest(5, 'churn_probability')[['customer_id', 'frequency', 'monetary', 'churn_probability']]
    for idx, row in top_risk.iterrows():
        print(f"  ID {row['customer_id']}: {row['churn_probability']*100:.1f}% (Freq: {row['frequency']}, Gasto: ${row['monetary']:.0f})")


def ejemplo_7_metricas_modelo():
    """Ejemplo 7: Analizar métricas del modelo"""
    print("\n" + "="*70)
    print("EJEMPLO 7: Métricas de Rendimiento del Modelo")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    metrics = predictor.metrics
    
    print("\n📈 Métricas principales:")
    print(f"  Accuracy (Train): {metrics['train_accuracy']*100:.2f}%")
    print(f"  Accuracy (Test):  {metrics['test_accuracy']*100:.2f}%")
    print(f"  Precision:        {metrics['test_precision']:.4f}")
    print(f"  Recall:           {metrics['test_recall']:.4f}")
    print(f"  F1-Score:         {metrics['test_f1']:.4f}")
    print(f"  AUC-ROC:          {metrics['test_auc']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\n📊 Matriz de Confusión:")
    print(f"  TN (correcto no-churn): {cm['tn']:5}")
    print(f"  FP (falso positivo):    {cm['fp']:5}")
    print(f"  FN (falso negativo):    {cm['fn']:5}")
    print(f"  TP (correcto churn):    {cm['tp']:5}")


def ejemplo_8_estrategia_retension():
    """Ejemplo 8: Estrategia de retención"""
    print("\n" + "="*70)
    print("EJEMPLO 8: Estrategia de Retención de Clientes")
    print("="*70)
    
    predictor = ChurnPredictor()
    predictor.train()
    
    # Crear base de clientes
    np.random.seed(42)
    customers = pd.DataFrame({
        'customer_id': range(1001, 1051),
        'frequency': np.random.poisson(8, 50),
        'monetary': np.random.normal(3000, 1500, 50),
        'avg_ticket': np.random.normal(500, 200, 50)
    })
    
    customers['frequency'] = customers['frequency'].clip(lower=1)
    customers['monetary'] = customers['monetary'].clip(lower=100)
    customers['avg_ticket'] = customers['avg_ticket'].clip(lower=50)
    
    # Predicción
    customers['churn_prob'] = predictor.predict_churn_probability(
        customers[['frequency', 'monetary', 'avg_ticket']]
    )
    
    # Estrategia
    print("\n💡 Recomendaciones de retención:")
    
    # Grupo 1: Alto riesgo, alto valor
    high_value_high_risk = customers[(customers['churn_prob'] > 0.7) & (customers['monetary'] > 5000)]
    print(f"\n1️⃣  PRIORIDAD ALTA - {len(high_value_high_risk)} clientes:")
    print("   Acción: Contacto personalizado, oferta especial")
    for _, c in high_value_high_risk.head(3).iterrows():
        print(f"   • Cliente {c['customer_id']}: Gasto=${c['monetary']:.0f}, Riesgo={c['churn_prob']*100:.0f}%")
    
    # Grupo 2: Bajo riesgo
    low_risk = customers[customers['churn_prob'] < 0.4]
    print(f"\n2️⃣  MANTENIMIENTO - {len(low_risk)} clientes:")
    print("   Acción: Comunicaciones regulares, programa de fidelización")
    
    # Grupo 3: Riesgo medio
    medium = customers[(customers['churn_prob'] >= 0.4) & (customers['churn_prob'] <= 0.7)]
    print(f"\n3️⃣  MONITOREO - {len(medium)} clientes:")
    print("   Acción: Seguimiento periódico, ofertas dinámicas")


def main():
    """Ejecutar todos los ejemplos"""
    print("\n" + "🚀 "*35)
    print("EJEMPLOS DE USO - CHURN PREDICTOR")
    print("🚀 "*35)
    
    try:
        # Descomentar los ejemplos que deseas ejecutar
        ejemplo_1_entrenamiento_basico()
        # ejemplo_2_cargar_modelo()
        # ejemplo_3_importancia_features()
        # ejemplo_4_prediccion_probabilidad()
        # ejemplo_5_clasificacion_binaria()
        # ejemplo_6_segmentacion_clientes()
        # ejemplo_7_metricas_modelo()
        # ejemplo_8_estrategia_retension()
        
        print("\n" + "="*70)
        print("✅ Ejemplos completados exitosamente")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
