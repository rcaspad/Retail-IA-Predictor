"""Module for loading and generating retail data from various sources."""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta


def generate_products(num_products: int = 5000) -> pd.DataFrame:
    """Generate synthetic product data.
    
    Args:
        num_products: Number of products to generate
        
    Returns:
        DataFrame with product data
    """
    categories = ['Hogar', 'Jardín', 'Construcción', 'Herramientas', 'Decoración']
    margins = np.random.uniform(0.2, 0.5, num_products)
    prices = np.random.uniform(5.0, 500.0, num_products)
    
    df = pd.DataFrame({
        'id': range(1, num_products + 1),
        'category': np.random.choice(categories, num_products),
        'price': np.round(prices, 2),
        'cost': np.round(prices * (1 - margins), 2)
    })
    
    return df


def generate_customers(num_customers: int = 50000) -> pd.DataFrame:
    """Generate synthetic customer data.
    
    Args:
        num_customers: Number of customers to generate
        
    Returns:
        DataFrame with customer data
    """
    segments = np.random.choice(
        ['Particular', 'Profesional', 'Empresa'],
        num_customers,
        p=[0.70, 0.25, 0.05]
    )
    
    churn_risk = np.random.uniform(0, 1, num_customers)
    
    df = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'segment': segments,
        'churn_risk': np.round(churn_risk, 3)
    })
    
    return df


def generate_transactions(
    num_transactions: int = 200000,
    num_customers: int = 50000,
    num_products: int = 5000,
    start_date: str = '2022-01-01'
) -> pd.DataFrame:
    """Generate synthetic transaction data with seasonality and trend.
    
    Args:
        num_transactions: Number of transactions to generate
        num_customers: Number of customers (for reference)
        num_products: Number of products (for reference)
        start_date: Start date for transactions
        
    Returns:
        DataFrame with transaction data
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime('2026-01-09')
    total_days = (end - start).days
    
    transactions = []
    
    # Calculate growth factor for trend (1% monthly = 1.01^month)
    # We'll apply a base growth rate that increases over months
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    for i in range(num_transactions):
        # Random date between start and end
        days_offset = random.randint(0, total_days)
        transaction_date = start + timedelta(days=days_offset)
        
        # Calculate months elapsed since start
        months_elapsed = (transaction_date.year - start_datetime.year) * 12 + \
                        (transaction_date.month - start_datetime.month)
        
        # Trend: 1% monthly cumulative growth
        trend_multiplier = 1.01 ** months_elapsed
        
        # Seasonality: Higher probability for 'Jardín' in summer months (6, 7, 8)
        month = transaction_date.month
        is_summer = month in [6, 7, 8]
        
        # Generate transaction with trend applied
        base_probability = 1.0
        adjusted_probability = base_probability * trend_multiplier
        
        # Adjust transaction likelihood based on seasonality
        if random.random() < adjusted_probability:
            # This transaction happens
            customer_id = random.randint(1, num_customers)
            product_id = random.randint(1, num_products)
            
            # Higher chance of selecting Jardín in summer
            if is_summer and random.random() < 0.5:
                # Override to a Jardín product (let's assume ids 1000-2000 are Jardín)
                product_id = random.randint(1000, min(2000, num_products))
            
            quantity = random.randint(1, 10)
            
            # Simulate price variation (±10%)
            base_price = 50.0  # Average price
            price_variation = base_price * np.random.uniform(0.9, 1.1)
            total_amount = np.round(quantity * price_variation, 2)
            
            transactions.append({
                'date': transaction_date.date(),
                'customer_id': customer_id,
                'product_id': product_id,
                'quantity': quantity,
                'total_amount': total_amount
            })
    
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    return pd.read_csv(filepath)


def load_data(data_dir: str = None) -> tuple:
    """Load data from raw data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of DataFrames (products, customers, transactions)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    products = load_csv(str(Path(data_dir) / 'products.csv'))
    customers = load_csv(str(Path(data_dir) / 'customers.csv'))
    transactions = load_csv(str(Path(data_dir) / 'transactions.csv'))
    
    return products, customers, transactions


def main():
    """Main function to generate and save synthetic retail data."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define data directory
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    print("Generating products data...")
    products = generate_products(num_products=5000)
    products.to_csv(data_dir / 'products.csv', index=False)
    print(f"  ✓ {len(products):,} productos guardados")
    
    print("Generating customers data...")
    customers = generate_customers(num_customers=50000)
    customers.to_csv(data_dir / 'customers.csv', index=False)
    print(f"  ✓ {len(customers):,} clientes guardados")
    
    print("Generating transactions data (con estacionalidad y tendencia)...")
    transactions = generate_transactions(
        num_transactions=200000,
        num_customers=50000,
        num_products=5000
    )
    transactions.to_csv(data_dir / 'transactions.csv', index=False)
    print(f"  ✓ {len(transactions):,} transacciones guardadas")
    
    # Summary statistics
    print("\n" + "="*60)
    print("RESUMEN DE DATOS GENERADOS")
    print("="*60)
    print(f"\nProductos: {len(products):,}")
    print(f"  Categorías: {products['category'].unique().tolist()}")
    print(f"  Precio promedio: ${products['price'].mean():.2f}")
    
    print(f"\nClientes: {len(customers):,}")
    print(f"  Segmentación:")
    for seg, count in customers['segment'].value_counts().items():
        print(f"    - {seg}: {count:,} ({count/len(customers)*100:.1f}%)")
    
    print(f"\nTransacciones: {len(transactions):,}")
    print(f"  Período: {transactions['date'].min().date()} a {transactions['date'].max().date()}")
    print(f"  Monto promedio: ${transactions['total_amount'].mean():.2f}")
    print(f"  Cantidad promedio: {transactions['quantity'].mean():.1f} unidades")
    
    print("\n" + "="*60)
    print("✅ Datos generados exitosamente en data/raw/")
    print("="*60)


if __name__ == "__main__":
    main()
