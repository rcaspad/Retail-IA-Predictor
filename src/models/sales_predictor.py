"""
Sales Time Series Predictor Module
Utiliza Prophet para predecir ventas diarias totales
"""

import pandas as pd
import pickle
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging

# Configurar logging para evitar advertencias innecesarias
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# Importar Prophet después de configurar logging
try:
    from prophet import Prophet
except ImportError:
    raise ImportError("Prophet no está instalado. Ejecuta: pip install prophet")

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class SalesTimeSeriesPredictor:
    """
    Clase para entrenar y predecir ventas diarias usando Prophet.
    
    Attributes:
        model (Prophet): Modelo Prophet entrenado
        data_path (str): Ruta del archivo de datos procesados
        model_path (str): Ruta donde guardar el modelo entrenado
        df_train (pd.DataFrame): Datos de entrenamiento en formato Prophet
    """
    
    def __init__(self, data_path='data/processed/sales_processed.csv', 
                 model_path='models/sales_model.pkl'):
        """
        Inicializa el predictor.
        
        Args:
            data_path (str): Ruta del archivo CSV con datos procesados
            model_path (str): Ruta donde guardar el modelo
        """
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.df_train = None
        
        # Crear directorio de modelos si no existe
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """
        Entrena el modelo Prophet con datos de ventas diarias.
        
        Pasos:
        1. Carga los datos procesados
        2. Agrupa por fecha sumando ventas diarias
        3. Renombra columnas al formato requerido por Prophet (ds, y)
        4. Entrena el modelo Prophet con seasonality anual
        5. Guarda el modelo en formato pickle
        
        Raises:
            FileNotFoundError: Si el archivo de datos no existe
            ValueError: Si no hay datos válidos para entrenar
        """
        print("📊 Iniciando entrenamiento del modelo de ventas...")
        
        # 1. Cargar datos
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Archivo no encontrado: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {len(df)} registros")
        
        # 2. Convertir fecha a datetime y agrupar por día
        df['date'] = pd.to_datetime(df['date'])
        
        df_daily = df.groupby('date').agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        print(f"✓ Serie temporal creada: {len(df_daily)} días")
        
        # 3. Renombrar columnas al formato Prophet (ds, y)
        self.df_train = df_daily.rename(columns={
            'date': 'ds',
            'total_amount': 'y'
        })[['ds', 'y']]
        
        # Validar datos
        if self.df_train.isnull().any().any():
            self.df_train = self.df_train.dropna()
            print(f"⚠ Valores nulos eliminados")
        
        if len(self.df_train) < 30:
            raise ValueError("❌ No hay suficientes datos para entrenar (mín. 30 días)")
        
        # 4. Instanciar y entrenar Prophet
        print("🔧 Configurando modelo Prophet...")
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
            seasonality_mode='additive'
        )
        
        print("📈 Entrenando modelo...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.df_train)
        
        # 5. Guardar modelo
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Modelo entrenado y guardado en: {self.model_path}")
        print(f"   Período de datos: {self.df_train['ds'].min().date()} a {self.df_train['ds'].max().date()}")
        print(f"   Ventas promedio diarias: ${self.df_train['y'].mean():.2f}")
    
    def predict_next_days(self, days=90):
        """
        Realiza predicciones para los próximos N días.
        
        Args:
            days (int): Número de días a predecir (default: 90)
        
        Returns:
            pd.DataFrame: DataFrame con predicciones (ds, yhat, yhat_lower, yhat_upper)
        
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
        """
        if self.model is None:
            raise RuntimeError("❌ El modelo no ha sido entrenado. Llama a train() primero.")
        
        # Generar fechas futuras
        future = self.model.make_future_dataframe(periods=days)
        
        # Realizar predicción
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        # Si df_train está disponible, retornar solo predicciones futuras
        if self.df_train is not None:
            future_forecast = forecast[forecast['ds'] > self.df_train['ds'].max()][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ].reset_index(drop=True)
        else:
            # Si no, retornar los últimos 'days' registros
            future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).reset_index(drop=True)
        
        return future_forecast
    
    def get_tomorrow_prediction(self):
        """
        Obtiene la predicción de ventas para mañana.
        
        Returns:
            dict: Diccionario con predicción (yhat, yhat_lower, yhat_upper)
        """
        if self.model is None:
            raise RuntimeError("❌ El modelo no ha sido entrenado.")
        
        # Si df_train no está disponible (modelo cargado), usar última fecha conocida
        # Generar predicción para 1 día
        future = self.model.make_future_dataframe(periods=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        # Obtener última predicción (mañana)
        if len(forecast) > 0:
            last_pred = forecast.iloc[-1]
            prediction = {
                'date': last_pred['ds'].date(),
                'yhat': last_pred['yhat'],
                'yhat_lower': last_pred['yhat_lower'],
                'yhat_upper': last_pred['yhat_upper']
            }
            return prediction
        
        return None
    
    def load_model(self):
        """
        Carga un modelo previamente entrenado.
        
        Raises:
            FileNotFoundError: Si el archivo del modelo no existe
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Modelo no encontrado: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✅ Modelo cargado desde: {self.model_path}")


if __name__ == "__main__":
    """
    Script principal para entrenar el modelo y realizar pruebas.
    """
    print("=" * 70)
    print("🚀 SALES TIME SERIES PREDICTOR - PROPHET MODEL")
    print("=" * 70 + "\n")
    
    # Crear instancia del predictor
    predictor = SalesTimeSeriesPredictor(
        data_path='data/processed/sales_processed.csv',
        model_path='models/sales_model.pkl'
    )
    
    # 1. Entrenar el modelo
    print("\n[1] FASE DE ENTRENAMIENTO")
    print("-" * 70)
    predictor.train()
    
    # 2. Realizar predicción de prueba para los próximos 30 días
    print("\n[2] PREDICCIÓN DE PRUEBA (Próximos 30 días)")
    print("-" * 70)
    forecast_30 = predictor.predict_next_days(days=30)
    print(f"✓ Predicción generada para 30 días")
    print(f"  Primeras 5 predicciones:")
    for idx, row in forecast_30.head().iterrows():
        print(f"  {row['ds'].date()}: ${row['yhat']:.2f} (rango: ${row['yhat_lower']:.2f} - ${row['yhat_upper']:.2f})")
    
    # 3. Obtener predicción para mañana
    print("\n[3] PREDICCIÓN PARA MAÑANA")
    print("-" * 70)
    tomorrow_pred = predictor.get_tomorrow_prediction()
    
    if tomorrow_pred:
        tomorrow_date = tomorrow_pred['date']
        tomorrow_sales = tomorrow_pred['yhat']
        lower_bound = tomorrow_pred['yhat_lower']
        upper_bound = tomorrow_pred['yhat_upper']
        
        # Validar predicción (valores no negativos)
        tomorrow_sales = max(0, tomorrow_sales)
        
        print(f"✅ Modelo de Ventas entrenado y guardado. Predicción mañana: ${tomorrow_sales:.2f}")
        print(f"   Fecha: {tomorrow_date}")
        print(f"   Intervalo de confianza (95%): ${lower_bound:.2f} - ${upper_bound:.2f}")
    else:
        print("❌ No se pudo generar predicción para mañana")
    
    print("\n" + "=" * 70)
    print("✨ Proceso completado exitosamente")
    print("=" * 70)
