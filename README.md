# Retail IA Predictor

Advanced Machine Learning system for retail sales forecasting and customer churn prediction.

## Project Structure

```
Retail-IA-Predictor/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and processed data
│   └── external/         # External data sources
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # ML models (sales, churn prediction)
│   ├── visualization/   # Visualization utilities
│   └── utils/           # Helper functions
├── app/                 # Streamlit application
│   └── pages/          # Application pages
├── models/             # Trained model artifacts
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Streamlit Application
```bash
streamlit run app/Home.py
```

### Use Models in Code
```python
from src.models.sales_predictor import SalesPredictor
from src.data.load_data import load_csv

# Load data
df = load_csv('data/raw/sales.csv')

# Create and train model
predictor = SalesPredictor()
predictor.train(X, y)

# Make predictions
predictions = predictor.predict(X_new)
```

## Features

- **Sales Prediction**: Forecast sales volume and revenue using ensemble methods
- **Churn Prediction**: Identify customers at risk of leaving
- **Interactive Dashboard**: Streamlit-based web interface
- **Data Processing**: Automated data cleaning and feature engineering
- **Model Tracking**: MLflow integration for experiment management

## Dependencies

Key libraries:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting
- **prophet**: Time series forecasting
- **plotly/altair**: Data visualization
- **streamlit**: Web application framework
- **fastapi**: API server
- **mlflow**: Model tracking and management

## Development

This project is organized as a Python package. To install in development mode:

```bash
pip install -e .
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue in the repository.
