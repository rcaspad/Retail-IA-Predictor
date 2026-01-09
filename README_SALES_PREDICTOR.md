# 🚀 Sales Time Series Predictor - README

## 📌 Descripción

Sistema de predicción de ventas diarias basado en **Facebook Prophet**, un modelo de series temporales especializado en datos de negocio. Predice ventas futuras con intervalos de confianza y captura patrones estacionales automáticamente.

## ✨ Características Clave

- 📊 **Predicción de Ventas Diarias**: Predice ventas futuras basado en datos históricos (2022-2026)
- 📈 **Análisis Estacional**: Captura patrones semanales y anuales automáticamente
- 📉 **Intervalos de Confianza**: Proporciona rangos de predicción al 95%
- 💾 **Persistencia**: Guarda el modelo entrenado para reutilización rápida
- 🔧 **Flexible**: Predice cualquier número de días (1 a N días)
- ✅ **Robusto**: Validación de datos, manejo de errores, logging detallado

## 🎯 Requisitos Previos

### Python 3.13+
```bash
python --version
```

### Dependencias
```bash
pip install -r requirements.txt
```

O instalar individuales:
```bash
pip install pandas==2.0.3 prophet==1.1.5 numpy==1.24.3
```

## 🗂️ Estructura del Proyecto

```
Retail-IA-Predictor/
├── src/
│   └── models/
│       └── sales_predictor.py          # ⭐ Clase SalesTimeSeriesPredictor
├── data/
│   └── processed/
│       ├── sales_processed.csv         # Datos de entrenamiento
│       └── forecast_30days.csv         # Salida de predicciones (opcional)
├── models/
│   └── sales_model.pkl                 # Modelo entrenado
├── requirements.txt                    # Dependencias
├── SALES_PREDICTOR_GUIDE.md           # Guía completa
├── IMPLEMENTATION_SUMMARY.md          # Resumen técnico
├── demo_load_model.py                 # Demo ejecutable
└── examples_sales_predictor.py        # Ejemplos de uso
```

## 🚀 Inicio Rápido

### 1. **Entrenar el Modelo (Opción 1)**

Ejecutar el script principal:
```bash
python src/models/sales_predictor.py
```

**Salida esperada:**
```
✅ Modelo de Ventas entrenado y guardado. Predicción mañana: $37383.47
```

### 2. **Usar el Modelo Preentrenado (Opción 2)**

```bash
python demo_load_model.py
```

### 3. **Usar en Tu Código**

```python
from src.models.sales_predictor import SalesTimeSeriesPredictor

# Cargar modelo preentrenado
predictor = SalesTimeSeriesPredictor()
predictor.load_model()

# Obtener predicción para mañana
tomorrow = predictor.get_tomorrow_prediction()
print(f"Ventas mañana: ${tomorrow['yhat']:.2f}")

# Predicción para próximos 30 días
forecast = predictor.predict_next_days(days=30)
print(forecast)
```

## 📚 Documentación Completa

- **[SALES_PREDICTOR_GUIDE.md](SALES_PREDICTOR_GUIDE.md)** - Guía de uso detallada
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Resumen técnico
- **[examples_sales_predictor.py](examples_sales_predictor.py)** - 8 ejemplos de código

## 🔑 Métodos Principales

### `train()`
Entrena el modelo con datos históricos y lo guarda en `models/sales_model.pkl`

```python
predictor = SalesTimeSeriesPredictor()
predictor.train()
```

### `predict_next_days(days=90)`
Predice ventas para N días en el futuro

```python
forecast = predictor.predict_next_days(days=30)
# Retorna DataFrame con columnas: ds, yhat, yhat_lower, yhat_upper
```

### `get_tomorrow_prediction()`
Obtiene predicción específica para mañana

```python
tomorrow = predictor.get_tomorrow_prediction()
# Retorna: {'date': YYYY-MM-DD, 'yhat': XXXX.XX, 'yhat_lower': ..., 'yhat_upper': ...}
```

### `load_model()`
Carga modelo preentrenado sin reentrenar

```python
predictor = SalesTimeSeriesPredictor()
predictor.load_model()
```

## 📊 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Registros entrenados** | 200,000 transacciones |
| **Período de datos** | 2022-01-01 a 2026-01-09 |
| **Días en modelo** | 1,470 días |
| **Ventas promedio diaria** | $37,401.87 |
| **Modelo guardado** | `models/sales_model.pkl` |

## 💡 Ejemplos de Predicción

```python
# Predicción para mañana
predictor = SalesTimeSeriesPredictor()
predictor.load_model()
tomorrow = predictor.get_tomorrow_prediction()

print(f"Fecha: {tomorrow['date']}")
print(f"Predicción: ${tomorrow['yhat']:.2f}")
print(f"Rango (95%): ${tomorrow['yhat_lower']:.2f} - ${tomorrow['yhat_upper']:.2f}")

# Salida:
# Fecha: 2026-01-10
# Predicción: $37383.47
# Rango (95%): $30655.92 - $44607.59
```

## 🔧 Configuración Avanzada

### Cambiar Rutas de Datos

```python
predictor = SalesTimeSeriesPredictor(
    data_path='mi_ruta/datos.csv',
    model_path='mi_ruta/modelo.pkl'
)
```

### Predecir Múltiples Períodos

```python
predictor.load_model()

# Próximos 7 días
forecast_7 = predictor.predict_next_days(days=7)

# Próximos 90 días
forecast_90 = predictor.predict_next_days(days=90)

# Analizar
print(f"Promedio (7d): ${forecast_7['yhat'].mean():.2f}")
print(f"Promedio (90d): ${forecast_90['yhat'].mean():.2f}")
```

## ⚙️ Modelo Prophet - Configuración

```python
Prophet(
    yearly_seasonality=True,      # Captura patrones anuales
    weekly_seasonality=True,      # Captura patrones semanales  
    daily_seasonality=False,      # No hay variación significativa
    interval_width=0.95,          # Intervalo de confianza 95%
    seasonality_mode='additive'   # Suma efectos estacionales
)
```

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'prophet'`
```bash
pip install prophet --upgrade
```

### Error: `FileNotFoundError: Archivo no encontrado`
Verificar que `data/processed/sales_processed.csv` existe

### Predicción muy lenta
- Primera ejecución: 30-60 segundos (normal, es entrenamiento)
- Predicciones posteriores: < 2 segundos

## 📋 Checklist de Uso

- [ ] Instalar requisitos: `pip install -r requirements.txt`
- [ ] Verificar datos en `data/processed/sales_processed.csv`
- [ ] Ejecutar demo: `python demo_load_model.py`
- [ ] Revisar ejemplos: `python examples_sales_predictor.py`
- [ ] Importar en tu proyecto: `from src.models.sales_predictor import SalesTimeSeriesPredictor`
- [ ] Cargar modelo: `predictor.load_model()`
- [ ] Hacer predicciones: `forecast = predictor.predict_next_days(days=30)`

## 🎓 Información Técnica

### ¿Por qué Prophet?

Facebook Prophet es la solución ideal para predicción de ventas retail porque:

1. **Maneja cambios de tendencia**: Detecta automáticamente puntos de cambio
2. **Estacionalidad múltiple**: Captura patrones semanales y anuales
3. **Robustez**: Funciona bien con datos reales (valores faltantes, outliers)
4. **Interpretable**: Proporciona componentes que se pueden analizar
5. **Rápido**: Predicciones muy rápidas después del entrenamiento
6. **Intervalos de confianza**: Proporciona rangos de predicción

### Fórmula Base

```
y_t = g(t) + s(t) + h(t) + ε_t

Donde:
- g(t) = Componente de tendencia
- s(t) = Componente de estacionalidad
- h(t) = Efectos de días especiales
- ε_t = Término de error
```

## 📞 Soporte

Para más información, consultar:
1. Docstrings en el código (`sales_predictor.py`)
2. `SALES_PREDICTOR_GUIDE.md` para guía detallada
3. `IMPLEMENTATION_SUMMARY.md` para información técnica
4. `examples_sales_predictor.py` para ejemplos de código

## 📝 Historial de Cambios

### v1.0.0 (2026-01-09) - Versión Inicial
- ✅ Implementación de clase `SalesTimeSeriesPredictor`
- ✅ Métodos: `train()`, `predict_next_days()`, `get_tomorrow_prediction()`, `load_model()`
- ✅ Modelo entrenado y guardado
- ✅ Documentación completa
- ✅ Ejemplos funcionales
- ✅ Demo ejecutable

## 📄 Licencia

Proyecto educativo - Libre para usar y modificar

---

**Última Actualización:** 2026-01-09  
**Versión:** 1.0.0  
**Estado:** ✅ Producción
