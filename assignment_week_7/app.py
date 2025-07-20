import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Optional: Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Function to load data
@st.cache_data # Cache data to prevent reloading on every rerun
def load_data(file_path):
    df = pd.read_csv(file_path)
    date_col = df.columns[df.columns.str.contains('date', case=False)].tolist()
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
        df.set_index(date_col[0], inplace=True)
    return df

# Function to train a model (example: RandomForestRegressor)
# We won't cache the model here with st.cache_resource directly if we're storing in session_state,
# as session_state itself effectively acts as a cache across reruns.
def train_model(df):
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return model, r2, mse

# Streamlit App
def main():
    st.title("Google Stock Price Prediction")

    st.write("This app demonstrates a machine learning model for Google stock price prediction based on historical data.")

    # Initialize session state variables if they don't exist
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'r2_score' not in st.session_state:
        st.session_state.r2_score = None
    if 'mse_score' not in st.session_state:
        st.session_state.mse_score = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None

    # File uploader for the dataset
    uploaded_file = st.file_uploader("Upload your GoogleStockPrices.csv file", type=["csv"])

    if uploaded_file is not None:
        # Load data only if it hasn't been loaded or a new file is uploaded
        if not st.session_state.data_loaded or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.data_loaded = True
            st.session_state.uploaded_file_name = uploaded_file.name # Store file name to detect changes
            st.session_state.model = None # Reset model if new data is loaded
            st.session_state.r2_score = None
            st.session_state.mse_score = None

        st.subheader("Raw Data Preview")
        st.write(st.session_state.df.head())

        # Train Model button
        if st.button("Train Model"):
            with st.spinner('Training model... This might take a moment.'):
                trained_model, r2, mse = train_model(st.session_state.df.copy())
                st.session_state.model = trained_model
                st.session_state.r2_score = r2
                st.session_state.mse_score = mse
                st.success("Model training complete!")
                # Force a rerun to display results and prediction interface immediately
                st.rerun()

        # Display model training results if available
        if st.session_state.model is not None:
            st.subheader("Model Training Results (Random Forest Regressor)")
            st.write(f"R-squared (R²): {st.session_state.r2_score:.4f}")
            st.write(f"Mean Squared Error (MSE): {st.session_state.mse_score:.4f}")

            st.subheader("Make a Prediction")
            st.write("Enter values for 'Open', 'High', 'Low', and 'Volume' to get a 'Close' price prediction.")

            # Get average values for initial display
            initial_open = float(st.session_state.df['Open'].mean())
            initial_high = float(st.session_state.df['High'].mean())
            initial_low = float(st.session_state.df['Low'].mean())
            initial_volume = float(st.session_state.df['Volume'].mean())


            col1, col2, col3, col4 = st.columns(4)
            with col1:
                open_price = st.number_input("Open Price", value=initial_open, key="open_input")
            with col2:
                high_price = st.number_input("High Price", value=initial_high, key="high_input")
            with col3:
                low_price = st.number_input("Low Price", value=initial_low, key="low_input")
            with col4:
                volume = st.number_input("Volume", value=initial_volume, key="volume_input")

            if st.button("Predict"):
                if st.session_state.model is not None:
                    input_data = pd.DataFrame([[open_price, high_price, low_price, volume]],
                                              columns=['Open', 'High', 'Low', 'Volume'])
                    prediction = st.session_state.model.predict(input_data)[0]
                    st.success(f"Predicted Close Price: **${prediction:.2f}**")
                else:
                    st.warning("Please train the model first by clicking 'Train Model'.")
    else:
        st.info("Please upload the `GoogleStockPrices.csv` file to begin.")

if __name__ == "__main__":
    main()