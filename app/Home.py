"""Main Streamlit application home page."""

import streamlit as st


def main():
    """Main home page."""
    st.set_page_config(page_title="Retail IA Predictor", layout="wide")
    
    st.title("🛍️ Retail IA Predictor")
    st.subtitle("Advanced Sales and Customer Churn Prediction System")
    
    st.markdown("""
    Welcome to the Retail IA Predictor application!
    
    This application provides:
    - **Sales Forecasting**: Predict future sales volume and revenue
    - **Churn Prediction**: Identify at-risk customers
    - **Business Intelligence**: Comprehensive analytics and insights
    
    Navigate through the pages using the sidebar to access different features.
    """)


if __name__ == "__main__":
    main()
