"""
Churn Prediction Model - Customer Abandonment Risk Detection
Utiliza XGBoost para identificar clientes en riesgo de abandono
"""

import pandas as pd
import pickle
import os
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ChurnPredictor:
    """
    Clase para entrenar y predecir abandono de clientes usando XGBoost.
    
    Attributes:
        model (XGBClassifier): Modelo XGBoost entrenado
        data_path (str): Ruta del archivo de datos procesados
        model_path (str): Ruta donde guardar el modelo entrenado
        feature_names (list): Nombres de las features utilizadas
        metrics (dict): Métricas de rendimiento del modelo
    """
    
    def __init__(self, data_path='data/processed/customer_features.csv',
                 model_path='models/churn_model.pkl',
                 random_state=42):
        """
        Inicializa el predictor de churn.
        
        Args:
            data_path (str): Ruta del archivo CSV con datos de clientes
            model_path (str): Ruta donde guardar el modelo
            random_state (int): Seed para reproducibilidad
        """
        self.data_path = data_path
        self.model_path = model_path
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.X_test = None
        self.y_test = None
        
        # Crear directorio de modelos si no existe
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    
    def _create_features(self, df):
        """
        Crea las features para el modelo.
        
        Pasos:
        1. Calcula 'avg_ticket' = monetary / frequency
        2. Define 'is_churn' como target (1 si recency > 90)
        
        Args:
            df (pd.DataFrame): DataFrame con datos de clientes
            
        Returns:
            tuple: (X, y, feature_names)
        """
        df_features = df.copy()
        
        # Calcular avg_ticket (ticket promedio por transacción)
        # Evitar división por cero
        df_features['avg_ticket'] = np.where(
            df_features['frequency'] > 0,
            df_features['monetary'] / df_features['frequency'],
            0
        )
        
        # Crear target: is_churn = 1 si recency > 90 días, 0 si no
        # Esto significa que el cliente no ha comprado en más de 90 días
        df_features['is_churn'] = (df_features['recency'] > 90).astype(int)
        
        # Features para el modelo
        # IMPORTANTE: No incluir 'recency' porque sería data leakage
        # (el target se define directamente de recency)
        feature_cols = ['frequency', 'monetary', 'avg_ticket']
        
        X = df_features[feature_cols]
        y = df_features['is_churn']
        
        return X, y, feature_cols
    
    def train(self, test_size=0.2):
        """
        Entrena el modelo XGBoost para predicción de churn.
        
        Pasos:
        1. Carga los datos
        2. Crea features y target
        3. Divide en train/test (80/20)
        4. Entrena XGBClassifier
        5. Calcula métricas
        6. Guarda el modelo
        
        Raises:
            FileNotFoundError: Si el archivo de datos no existe
            ValueError: Si hay insuficientes datos
        """
        print("📊 Iniciando entrenamiento del modelo de Churn...")
        
        # 1. Cargar datos
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Archivo no encontrado: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {len(df)} clientes")
        
        # 2. Crear features y target
        print("🔧 Creando features...")
        X, y, feature_names = self._create_features(df)
        self.feature_names = feature_names
        
        # Información sobre el dataset
        n_churn = (y == 1).sum()
        n_no_churn = (y == 0).sum()
        churn_rate = (n_churn / len(y)) * 100
        
        print(f"✓ Features creadas: {len(feature_names)} features")
        print(f"  - Clientes sin riesgo (recency ≤ 90): {n_no_churn} ({100-churn_rate:.1f}%)")
        print(f"  - Clientes en riesgo (recency > 90): {n_churn} ({churn_rate:.1f}%)")
        
        # 3. Dividir en train/test (80/20)
        print("📋 Dividiendo en train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Mantener proporción de clases
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"✓ Train set: {len(X_train)} muestras")
        print(f"✓ Test set: {len(X_test)} muestras")
        
        # 4. Entrenar XGBClassifier
        print("🎓 Entrenando modelo XGBoost...")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0,
            eval_metric='logloss'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
        
        # 5. Calcular métricas
        print("📈 Calculando métricas...")
        self._calculate_metrics(X_train, y_train, X_test, y_test)
        
        # 6. Guardar modelo
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Modelo entrenado y guardado en: {self.model_path}")
    
    def _calculate_metrics(self, X_train, y_train, X_test, y_test):
        """
        Calcula y almacena métricas de rendimiento.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
        """
        # Predicciones
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas de entrenamiento
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Métricas de prueba
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        # Guardar métricas
        self.metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        # Mostrar resultados
        print("\n" + "="*70)
        print("📊 CLASIFICACIÓN REPORT - PREDICCIÓN DE CHURN")
        print("="*70)
        print(f"\nAccuracy (Train): {train_accuracy*100:.2f}%")
        print(f"Accuracy (Test):  {test_accuracy*100:.2f}%")
        print(f"\nMetricas de Test:")
        print(f"  Precision: {test_precision:.4f} (de los predichos como churn, cuántos realmente lo son)")
        print(f"  Recall:    {test_recall:.4f} (de los clientes en churn, cuántos identificamos)")
        print(f"  F1-Score:  {test_f1:.4f} (balance entre precision y recall)")
        print(f"  AUC-ROC:   {test_auc:.4f} (capacidad discriminativa del modelo)")
        
        print(f"\nMatriz de Confusión:")
        print(f"  True Negatives:  {tn} (correctamente identificados como NO en riesgo)")
        print(f"  False Positives: {fp} (incorrectamente marcados como en riesgo)")
        print(f"  False Negatives: {fn} (clientes en riesgo que no detectamos)")
        print(f"  True Positives:  {tp} (correctamente identificados como en riesgo)")
        
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=['No Churn', 'Churn'],
            digits=4
        ))
        print("="*70)
    
    def predict_churn_probability(self, X):
        """
        Predice la probabilidad de churn para nuevos clientes.
        
        Args:
            X (pd.DataFrame): Features de clientes
            
        Returns:
            np.ndarray: Probabilidades de churn (0 a 1)
        """
        if self.model is None:
            raise RuntimeError("❌ El modelo no ha sido entrenado.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """
        Predice si un cliente está en riesgo de churn (clasificación binaria).
        
        Args:
            X (pd.DataFrame): Features de clientes
            
        Returns:
            np.ndarray: Predicciones (0 = No churn, 1 = Churn)
        """
        if self.model is None:
            raise RuntimeError("❌ El modelo no ha sido entrenado.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        Obtiene la importancia de las features.
        
        Returns:
            pd.DataFrame: Features ordenadas por importancia
        """
        if self.model is None:
            raise RuntimeError("❌ El modelo no ha sido entrenado.")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
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
        
        # Inicializar feature_names si no están disponibles
        if self.feature_names is None:
            self.feature_names = ['frequency', 'monetary', 'avg_ticket']
        
        print(f"✅ Modelo cargado desde: {self.model_path}")


if __name__ == "__main__":
    """
    Script principal para entrenar el modelo de churn.
    """
    print("="*70)
    print("🚀 CHURN PREDICTOR - XGBOOST CLASSIFICATION MODEL")
    print("="*70 + "\n")
    
    # Crear instancia del predictor
    predictor = ChurnPredictor(
        data_path='data/processed/customer_features.csv',
        model_path='models/churn_model.pkl'
    )
    
    # Entrenar el modelo
    print("[1] FASE DE ENTRENAMIENTO")
    print("-"*70)
    predictor.train(test_size=0.2)
    
    # Mostrar importancia de features
    print("\n[2] IMPORTANCIA DE FEATURES")
    print("-"*70)
    feature_importance = predictor.get_feature_importance()
    print("\nImportancia de las features para predicción de churn:")
    for idx, row in feature_importance.iterrows():
        bar_length = int(row['importance'] * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {row['feature']:15} | {bar} {row['importance']:.4f}")
    
    # Información final
    print("\n" + "="*70)
    accuracy_pct = predictor.metrics['test_accuracy'] * 100
    print(f"✅ Modelo de Churn entrenado. Accuracy: {accuracy_pct:.2f}%")
    print("="*70)
