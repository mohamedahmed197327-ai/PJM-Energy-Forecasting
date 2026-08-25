import streamlit as st
import pandas as pd
import xgboost as xgb
import datetime

# Page configuration
st.set_page_config(page_title="Load Forecasting", page_icon="⚡", layout="centered")

st.title("⚡ PJM Smart Grid Load Forecasting App")
st.write("Enter a future date and time to predict the expected electricity demand in Megawatts (MW).")

# 1. Load the saved model
@st.cache_resource 
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('pjme_xgboost_model.json')
    return model

model = load_model()

# 2. User Input UI
st.markdown("### 📅 Select Prediction Time:")
col1, col2 = st.columns(2)

with col1:
    user_date = st.date_input("Select Date", datetime.date(2018, 8, 1))
with col2:
    user_time = st.time_input("Select Time (Hour)", datetime.time(12, 0))

# 3. Prediction Button
if st.button("Predict Demand 💡", use_container_width=True):
    # Combine date and time
    dt = datetime.datetime.combine(user_date, user_time)
    
    # Extract temporal features just like the training data
    input_features = pd.DataFrame({
        'hour': [dt.hour],
        'dayofweek': [dt.weekday()],
        'quarter': [(dt.month - 1) // 3 + 1],
        'month': [dt.month],
        'year': [dt.year],
        'dayofyear': [dt.timetuple().tm_yday],
        'dayofmonth': [dt.day],
        'weekofyear': [dt.isocalendar().week]
    })
    
    # Predict using the model
    prediction = model.predict(input_features)[0]
    
    # Display the result
    st.success(f"📈 **Expected Demand:** {prediction:,.2f} MW")
    st.info("This prediction is calculated based on the XGBoost algorithm and historical consumption patterns.")