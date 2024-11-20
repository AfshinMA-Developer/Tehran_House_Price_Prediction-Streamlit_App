import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from utils import data_cleaning

# Load Dataset and Models
DATASET_PATH = "datasets/housePrice.csv"
MODELS_PATH = [
    "models/KernelRidge_pipeline.joblib",
    "models/GradientBoostingRegressor_pipeline.joblib",
    "models/XGBoostRegressor_pipeline.joblib",
    "models/CatBoostRegressor_pipeline.joblib",
]

# Load the cleaned data
df = data_cleaning(DATASET_PATH)

# Prepare data for input fields
min_area, max_area = df['Area'].min(), df['Area'].max()
rooms = df['Room'].unique().tolist()
addresses = df['Address'].unique().tolist()

# Streamlit app layout
st.title("🏠 Tehran House Price Prediction")
st.sidebar.header("Input Parameters")

# Sidebar for input fields
st.sidebar.subheader("Enter the details:")
area = st.sidebar.number_input("Area (m²)", min_value=min_area, max_value=max_area, value=min_area, step=10)
room = st.sidebar.selectbox("Room", options=rooms)
parking = st.sidebar.checkbox("Parking", value=True)
warehouse = st.sidebar.checkbox("Warehouse", value=True)
elevator = st.sidebar.checkbox("Elevator", value=True)
address = st.sidebar.selectbox("Address", options=addresses)

# Prepare input data for prediction
sample = pd.DataFrame({
    'Area': [area],
    'Room': [room],
    'Parking': [parking],
    'Warehouse': [warehouse],
    'Elevator': [elevator],
    'Address': [address]
})

def load_and_predict(sample):
    result = {
        'Model': [],
        'R2': [],
        'Predicted_Price_(IRR)': []
    }

    # Define features and target variable
    X = df.drop(columns=['Price'])  # Features
    y = df['Price']

    try:
        for path in MODELS_PATH:
            model_name = path.split('/')[-1].split('_')[0]
            model = joblib.load(path)  # Load the model once

            # Predict house price
            y_pred = model.predict(X)
            price_pred = model.predict(sample)[0]

            result['Model'].append(model_name)
            result['R2'].append(r2_score(y, y_pred))
            result['Predicted_Price_(IRR)'].append(price_pred)

    except Exception as e:
        st.error(f"An error occurred during model loading or prediction: {str(e)}")
        return None
    return pd.DataFrame(result).sort_values(by=['R2'], ascending=False)

# Predict button
if st.sidebar.button("Predict"):
    result_df = load_and_predict(sample)

    if result_df is not None:
        st.success('Predicted House Price:')
        st.table(result_df)

# Footer or additional information
st.sidebar.markdown("### About this App")
st.sidebar.markdown(
    "This app predicts house prices based on input features such as area, number of rooms, "
    "and facilities like parking, warehouse, and elevator. Please fill in all fields to get the prediction."
)
