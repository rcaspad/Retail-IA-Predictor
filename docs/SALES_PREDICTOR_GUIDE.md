# 📊 Sales Time Series Predictor - Guía de Uso

## Descripción General

`SalesTimeSeriesPredictor` es una clase especializada en Machine Learning que utiliza **Facebook Prophet** para predecir ventas diarias totales basándose en datos históricos.

## 🎯 Características

- ✅ Carga automática de datos procesados
- ✅ Agregación diaria de ventas (suma de `total_amount`)
- ✅ Entrenamiento de modelo Prophet con seasonality anual
- ✅ Predicciones para N días en el futuro
- ✅ Intervalos de confianza (95%)
- ✅ Persistencia del modelo (guardado en pickle)
- ✅ Manejo robusto de errores

## 📦 Requisitos

```
pandas>=2.0.3
prophet>=1.1.5
numpy>=1.24.3
```

Instalar todos los requisitos:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### 1. Entrenamiento del Modelo

```python
from src.models.sales_predictor import SalesTimeSeriesPredictor

# Crear instancia
predictor = SalesTimeSeriesPredictor(
    data_path='data/processed/sales_processed.csv',
    model_path='models/sales_model.pkl'
)

# Entrenar el modelo
predictor.train()
```

### 2. Predicción para Próximos Días

```python
# Predecir ventas para los próximos 90 días
forecast = predictor.predict_next_days(days=90)

# Ver las predicciones
print(forecast.head())
# Columnas: ds (fecha), yhat (predicción), yhat_lower, yhat_upper
```

### 3. Predicción para Mañana

```python
# Obtener predicción específica para mañana
tomorrow = predictor.get_tomorrow_prediction()

print(f"Predicción para {tomorrow['date']}")
print(f"Ventas esperadas: ${tomorrow['yhat']:.2f}")
print(f"Rango (95%): ${tomorrow['yhat_lower']:.2f} - ${tomorrow['yhat_upper']:.2f}")
```

### 4. Cargar un Modelo Preentrenado

```python
# Crear nueva instancia
predictor = SalesTimeSeriesPredictor()

# Cargar modelo existente
predictor.load_model()

# Realizar predicciones sin reentrenar
forecast = predictor.predict_next_days(days=30)
```

## 📈 Estructura de Datos

### Entrada
Archivo: `data/processed/sales_processed.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | datetime | Fecha de la transacción |
| `total_amount` | float | Monto total de la venta |
| `quantity` | int | Cantidad de productos |

### Salida (Predicción)

DataFrame con columnas:
- **ds**: Fecha predicha (datetime)
- **yhat**: Predicción puntual de ventas (float)
- **yhat_lower**: Límite inferior del intervalo de confianza (float)
- **yhat_upper**: Límite superior del intervalo de confianza (float)

## 🔧 Parámetros del Modelo

El modelo Prophet está configurado con:
- `yearly_seasonality=True`: Captura patrones anuales
- `weekly_seasonality=True`: Captura patrones semanales
- `daily_seasonality=False`: No hay variación diaria significativa
- `seasonality_mode='additive'`: Suma efectos estacionales
- `interval_width=0.95`: Intervalo de confianza del 95%

## 📊 Ejemplo Completo

```python
from src.models.sales_predictor import SalesTimeSeriesPredictor

# Inicializar
predictor = SalesTimeSeriesPredictor()

# Entrenar
predictor.train()

# Predecir próximos 30 días
forecast_30 = predictor.predict_next_days(days=30)

# Analizar predicción para mañana
tomorrow = predictor.get_tomorrow_prediction()
print(f"Ventas predichas para mañana: ${tomorrow['yhat']:.2f}")

# Obtener promedio de predicción para próximos 7 días
forecast_7 = predictor.predict_next_days(days=7)
avg_sales_7days = forecast_7['yhat'].mean()
print(f"Promedio de ventas (próximos 7 días): ${avg_sales_7days:.2f}")
```

## ⚠️ Notas Importantes

1. **Datos Históricos**: El modelo requiere al menos 30 días de datos históricos
2. **Entrenamiento**: El primer entrenamiento puede tomar 30-60 segundos
3. **Precisión**: La precisión mejora con más datos históricos
4. **Estacionalidad**: El modelo captura patrones semanales y anuales
5. **Valores Negativos**: Las predicciones se validan para no ser negativas

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'prophet'`
```bash
pip install prophet
```

### Error: `FileNotFoundError: Archivo no encontrado`
Verificar que los datos procesados existen en `data/processed/sales_processed.csv`

### Predicción Lenta
- Primera predicción es lenta (30-60s) - es normal
- Predicciones posteriores son mucho más rápidas

## 📝 Historial de Datos

**Entrenamiento Exitoso:**
- Período: 2022-01-01 a 2026-01-09 (1470 días)
- Registros: 200,000 transacciones
- Ventas promedio diarias: $37,401.87
- Modelo guardado: `models/sales_model.pkl`

## 🎓 Información Técnica

### ¿Por qué Prophet?

Facebook Prophet es ideal para este caso de uso porque:
- Maneja bien cambios de tendencias
- Captura estacionalidad (semanal, anual)
- Robusto ante datos faltantes
- Proporciona intervalos de confianza
- Funciona bien con series temporales de retail

### Fórmula Base
```
y_t = g(t) + s(t) + h(t) + ε_t
```

Donde:
- `g(t)`: Componente de tendencia
- `s(t)`: Componente de estacionalidad
- `h(t)`: Efectos de días festivos
- `ε_t`: Término de error

---

**Última Actualización:** 2026-01-09
**Versión:** 1.0.0
