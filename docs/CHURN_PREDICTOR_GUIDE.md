# 📊 Churn Predictor - Guía de Uso

## Descripción General

`ChurnPredictor` es una clase especializada en Machine Learning para **identificar clientes en riesgo de abandono** usando **XGBoost**, uno de los algoritmos de clasificación más poderosos en la industria.

## 🎯 Características

- ✅ Detección automática de clientes en riesgo
- ✅ Definición de churn basada en recency (>90 días sin compra)
- ✅ Features robustas (frequency, monetary, avg_ticket)
- ✅ Prevención de data leakage
- ✅ Métricas detalladas de clasificación
- ✅ Importancia de features explicable
- ✅ Modelo guardado en pickle para producción

## 📦 Requisitos

```
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.3
numpy>=1.24.3
```

Instalar:
```bash
pip install xgboost scikit-learn
```

## 🚀 Uso Rápido

### 1. Entrenar el Modelo

```python
from src.models.churn_predictor import ChurnPredictor

# Crear y entrenar
predictor = ChurnPredictor()
predictor.train(test_size=0.2)
```

### 2. Cargar Modelo Preentrenado

```python
predictor = ChurnPredictor()
predictor.load_model()
```

### 3. Hacer Predicciones

```python
# Probabilidad de churn
probabilities = predictor.predict_churn_probability(X)

# Clasificación binaria (0/1)
predictions = predictor.predict(X)

# Importancia de features
importance = predictor.get_feature_importance()
```

## 📈 Definición de Churn

```
is_churn = 1  si  recency > 90 días  (sin compras por >3 meses)
is_churn = 0  si  recency ≤ 90 días  (cliente activo)
```

## 🔑 Features Utilizadas

| Feature | Descripción | Cálculo |
|---------|-------------|---------|
| **frequency** | Número de compras del cliente | Dato directo |
| **monetary** | Valor total gastado | Dato directo |
| **avg_ticket** | Ticket promedio | monetary / frequency |

### ⚠️ Por qué NO usamos recency

- `recency` es el **tiempo desde la última compra**
- El target `is_churn` se define directamente de `recency`
- Usar `recency` como input causaría **data leakage**
- El modelo aprendería a predecir el target trivialmente
- Por eso solo usamos comportamiento de gasto (frequency, monetary)

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Clientes entrenados** | 49,118 |
| **Churn rate** | 77.9% |
| **Train accuracy** | 78.11% |
| **Test accuracy** | 77.79% |
| **Precision** | 0.7802 |
| **Recall** | 0.9950 |
| **F1-Score** | 0.8746 |
| **AUC-ROC** | 0.6484 |

### Explicación de Métricas

- **Accuracy**: Porcentaje general de predicciones correctas
- **Precision**: De los predichos como churn, cuántos realmente lo son
- **Recall**: De los clientes realmente en churn, cuántos identificamos (99.5% ✅)
- **F1-Score**: Balance entre precision y recall
- **AUC-ROC**: Capacidad del modelo para discriminar entre clases

## 💡 Importancia de Features

```
frequency:   59.51% ████████████████████ 
monetary:    21.71% ███████
avg_ticket:  18.78% ██████
```

**Interpretación**: La frecuencia de compra es el predictor más importante del abandono. Clientes que compran con baja frecuencia tienen mayor riesgo de churn.

## 🔧 Métodos Principales

### `train(test_size=0.2)`
Entrena el modelo con datos de `data/processed/customer_features.csv`

```python
predictor = ChurnPredictor()
predictor.train()  # Split 80/20 automático
```

### `predict(X)`
Predice si clientes están en riesgo (0 o 1)

```python
predictions = predictor.predict(new_customers_df)
# Retorna: array([1, 0, 1, ...])
```

### `predict_churn_probability(X)`
Predice probabilidad de churn (0-1)

```python
probabilities = predictor.predict_churn_probability(new_customers_df)
# Retorna: array([0.95, 0.23, 0.87, ...])
```

### `get_feature_importance()`
Obtiene importancia de features

```python
importance = predictor.get_feature_importance()
print(importance)
#     feature  importance
# 0  frequency     0.5951
# 1  monetary      0.2171
# 2  avg_ticket    0.1878
```

### `load_model()`
Carga modelo preentrenado

```python
predictor = ChurnPredictor()
predictor.load_model()
```

## 📋 Matriz de Confusión Explicada

```
                 Predicho No-Churn    Predicho Churn
Real No-Churn            31              2,144
Real Churn               38              7,611
```

- **TN = 31**: Correctamente identificamos 31 clientes sin riesgo
- **FP = 2,144**: Falsos positivos (dijimos churn pero no lo eran)
- **FN = 38**: Falsos negativos (dijimos no-churn pero sí lo eran) ← Problema
- **TP = 7,611**: Correctamente identificamos 7,611 en riesgo

**Análisis**: El modelo es muy conservador - predice casi todos como churn. Esto es mejor que los falsos negativos (clientes abandonados no detectados).

## 🎓 Configuración de XGBoost

```python
XGBClassifier(
    n_estimators=100,        # 100 árboles
    max_depth=6,             # Profundidad máxima
    learning_rate=0.1,       # Tasa de aprendizaje
    subsample=0.8,           # 80% de datos por árbol
    colsample_bytree=0.8,    # 80% de features por árbol
    eval_metric='logloss'
)
```

## 💼 Casos de Uso

### 1. Identificar Clientes en Riesgo
```python
predictor.load_model()
probs = predictor.predict_churn_probability(customer_data)

# Clientes con riesgo alto (>80%)
high_risk = customer_data[probs > 0.8]
```

### 2. Propensity Scoring
```python
scores = predictor.predict_churn_probability(all_customers)
customers['churn_risk_score'] = scores

# Segmentar por riesgo
low_risk = customers[scores < 0.3]
medium_risk = customers[(scores >= 0.3) & (scores < 0.7)]
high_risk = customers[scores >= 0.7]
```

### 3. Campaña de Retención
```python
# Clientes recientes a retener
to_retain = high_risk[high_risk['frequency'] > 3]
print(f"Enviar ofertas a {len(to_retain)} clientes")
```

## ⚠️ Limitaciones

1. **Recency > 90 días**: Es una definición simple de churn
2. **No considera contexto**: Industria, estacionalidad, etc.
3. **Data labeling**: Usa recency como proxy de churn
4. **Clase desbalanceada**: 77.9% churn vs 22.1% activos
5. **Require actualización**: El modelo se vuelve obsoleto con el tiempo

## 🔄 Mejoras Posibles

- Incorporar features temporales (seasonality)
- Crear múltiples modelos por segmento
- Usar threshold dinámico según costo de FP/FN
- Implementar reentrenamiento automático
- Agregar features de comportamiento web/app

## 📞 Troubleshooting

### Error: `ModuleNotFoundError: xgboost`
```bash
pip install xgboost
```

### Error: `FileNotFoundError: customer_features.csv`
```bash
# Asegúrate que el archivo existe en:
data/processed/customer_features.csv
```

### Accuracy bajo
- Revisar data quality
- Ajustar threshold de recency (>90)
- Reentrenar con datos más recientes

## 📚 Archivos Relacionados

- [CHURN_PREDICTOR_EXAMPLES.md](CHURN_PREDICTOR_EXAMPLES.md) - Ejemplos de código
- [README_SALES_PREDICTOR.md](README_SALES_PREDICTOR.md) - Modelo de ventas
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Resumen técnico

---

**Última Actualización:** 2026-01-09  
**Versión:** 1.0.0  
**Accuracy:** 77.79%
