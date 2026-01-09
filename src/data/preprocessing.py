"""Module for data preprocessing and feature engineering."""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Pipeline de procesamiento de datos para análisis de retail."""
    
    def __init__(self, raw_data_path: str = 'data/raw', processed_data_path: str = 'data/processed'):
        """
        Inicializa el preprocessor.
        
        Args:
            raw_data_path: Ruta a los datos crudos
            processed_data_path: Ruta donde guardar datos procesados
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # DataFrames
        self.customers = None
        self.products = None
        self.transactions = None
        self.sales_processed = None
        self.customer_features = None
        
    def load_data(self):
        """Carga los datos desde archivos CSV."""
        print("📂 Cargando datos...")
        
        self.customers = pd.read_csv(self.raw_data_path / 'customers.csv')
        self.products = pd.read_csv(self.raw_data_path / 'products.csv')
        self.transactions = pd.read_csv(self.raw_data_path / 'transactions.csv')
        
        # Renombrar columna 'id' a 'product_id' en productos para hacer match con transacciones
        self.products.rename(columns={'id': 'product_id'}, inplace=True)
        
        print(f"  ✓ Clientes: {len(self.customers)} registros")
        print(f"  ✓ Productos: {len(self.products)} registros")
        print(f"  ✓ Transacciones: {len(self.transactions)} registros")
        
    def preprocess_transactions(self):
        """
        Procesa las transacciones:
        - Convierte fecha a datetime
        - Crea columnas temporales
        - Enriquece con información de productos
        - Calcula margen
        """
        print("\n🔧 Procesando transacciones...")
        
        # Convertir fecha a datetime
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        
        # Crear columnas temporales
        self.transactions['year'] = self.transactions['date'].dt.year
        self.transactions['month'] = self.transactions['date'].dt.month
        self.transactions['day_of_week'] = self.transactions['date'].dt.dayofweek
        
        # Merge con productos para obtener categoría y costo
        self.sales_processed = self.transactions.merge(
            self.products[['product_id', 'category', 'price', 'cost']],
            on='product_id',
            how='left'
        )
        
        # Calcular margen
        self.sales_processed['margin'] = (
            (self.sales_processed['price'] - self.sales_processed['cost']) * 
            self.sales_processed['quantity']
        )
        
        print(f"  ✓ Transacciones enriquecidas: {len(self.sales_processed)} registros")
        print(f"  ✓ Nuevas columnas: year, month, day_of_week, category, cost, margin")
        
    def create_customer_features(self):
        """
        Crea características RFM (Recency, Frequency, Monetary) por cliente.
        
        - Recency: Días desde la última compra
        - Frequency: Número de transacciones
        - Monetary: Total gastado
        """
        print("\n📊 Creando características de clientes (RFM)...")
        
        # Obtener la fecha máxima del dataset como referencia
        max_date = self.sales_processed['date'].max()
        
        # Calcular RFM por cliente
        rfm = self.sales_processed.groupby('customer_id').agg({
            'date': lambda x: (max_date - x.max()).days,  # Recency
            'product_id': 'count',  # Frequency
            'total_amount': 'sum'  # Monetary
        }).reset_index()
        
        # Renombrar columnas
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Merge con información de clientes
        self.customer_features = rfm.merge(
            self.customers,
            on='customer_id',
            how='left'
        )
        
        print(f"  ✓ Clientes procesados: {len(self.customer_features)} registros")
        print(f"  ✓ Métricas RFM calculadas:")
        print(f"    - Recency promedio: {self.customer_features['recency'].mean():.1f} días")
        print(f"    - Frequency promedio: {self.customer_features['frequency'].mean():.1f} transacciones")
        print(f"    - Monetary promedio: ${self.customer_features['monetary'].mean():.2f}")
        
    def save_data(self):
        """Guarda los datos procesados en archivos CSV."""
        print("\n💾 Guardando datos procesados...")
        
        # Crear directorio si no existe
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar transacciones procesadas
        sales_path = self.processed_data_path / 'sales_processed.csv'
        self.sales_processed.to_csv(sales_path, index=False)
        print(f"  ✓ Guardado: {sales_path}")
        
        # Guardar características de clientes
        customers_path = self.processed_data_path / 'customer_features.csv'
        self.customer_features.to_csv(customers_path, index=False)
        print(f"  ✓ Guardado: {customers_path}")
        
    def run_pipeline(self):
        """Ejecuta el pipeline completo de procesamiento."""
        self.load_data()
        self.preprocess_transactions()
        self.create_customer_features()
        self.save_data()
        
        print(f"\n✅ Datos procesados guardados: {len(self.sales_processed)} transacciones y {len(self.customer_features)} clientes.")


if __name__ == "__main__":
    # Ejecutar el pipeline de procesamiento
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()
