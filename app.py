import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 1. Page Configuration
st.set_page_config(page_title="PJME Load Forecasting", layout="wide")
st.title("⚡ Smart Grid Load Forecasting (PJME)")
st.write("Interactive application for hourly electricity demand forecasting using XGBoost.")

# 2. Define features in the exact same order used during training
FEATURES = [
    'hour', 'dayofweek', 'quarter', 'month', 'year',
    'dayofyear', 'dayofmonth', 'weekofyear',
    'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_mean_24h', 'rolling_mean_168h',
]

# 3. Load Model and Data functions
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model('pjme_xgboost_model.json')
    return model

@st.cache_data
def load_data():
    df = pd.read_csv('PJME_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime', 'PJME_MW'])
    df = df.drop_duplicates('Datetime').sort_values('Datetime').set_index('Datetime')
    return df

# 4. Recursive Forecast Function
def recursive_forecast(model, history_series, features, n_hours=24):
    history = history_series.copy()
    predictions = []
    last_timestamp = history.index.max()

    for i in range(1, n_hours + 1):
        next_timestamp = last_timestamp + pd.Timedelta(hours=i)
        
        row = {
            'hour': next_timestamp.hour,
            'dayofweek': next_timestamp.dayofweek,
            'quarter': next_timestamp.quarter,
            'month': next_timestamp.month,
            'year': next_timestamp.year,
            'dayofyear': next_timestamp.dayofyear,
            'dayofmonth': next_timestamp.day,
            'weekofyear': int(next_timestamp.isocalendar().week),
            'lag_24h': history.iloc[-24],
            'lag_48h': history.iloc[-48],
            'lag_168h': history.iloc[-168],
            'rolling_mean_24h': history.iloc[-24:].mean(),
            'rolling_mean_168h': history.iloc[-168:].mean(),
        }

        X_next = pd.DataFrame([row])[features]
        pred = model.predict(X_next)[0]
        
        predictions.append({'Datetime': next_timestamp, 'Predicted_MW': pred})
        history.loc[next_timestamp] = pred

    return pd.DataFrame(predictions).set_index('Datetime')

# 5. Build the User Interface (UI)
try:
    model = load_model()
    df = load_data()
    st.success("Model and data loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model or data. Details: {e}")
    st.stop()

# --- إعدادات اختيار التاريخ ---
last_available_date = df.index.max().date()
first_available_date = df.index.min().date()
max_allowed_date = last_available_date + pd.Timedelta(days=30) 

st.sidebar.header("⚙️ Forecast Settings")
st.sidebar.write(f"**Historical Data Range:**\n{first_available_date} to {last_available_date}")

target_date = st.sidebar.date_input(
    "Select Target Date:",
    value=last_available_date + pd.Timedelta(days=1),
    min_value=first_available_date, 
    max_value=max_allowed_date
)

if st.sidebar.button("Get Forecast 🚀"):
    
    with st.spinner(f'Processing data for {target_date}...'):
        
        # --- الحالة الأولى: لو المستخدم اختار تاريخ في الماضي ---
        if target_date <= last_available_date:
            target_data = df[df.index.date == target_date].copy()
            
            if target_data.empty:
                st.warning("No data available for this specific date.")
            else:
                predictions = model.predict(target_data[FEATURES])
                target_data['Predicted_MW'] = predictions
                
                st.subheader(f"Historical Evaluation: {target_date}")
                
                # عرض الإحصائيات جنب بعض
                col1, col2 = st.columns(2)
                with col1:
                    avg_load = target_data['Predicted_MW'].mean()
                    st.metric("Avg Predicted Load (MW)", f"{avg_load:,.2f}")
                with col2:
                    max_load = target_data['Predicted_MW'].max()
                    st.metric("Max Predicted Peak (MW)", f"{max_load:,.2f}")
                
                st.write("---")
                # عرض الجدول بعرض الشاشة
                display_df = target_data[['PJME_MW', 'Predicted_MW']].rename(columns={'PJME_MW': 'Actual_MW'})
                st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)

        # --- الحالة الثانية: لو المستخدم اختار تاريخ في المستقبل ---
        else:
            target_end_time = pd.to_datetime(f"{target_date} 23:00:00")
            hours_to_predict = int((target_end_time - df.index.max()).total_seconds() / 3600)

            full_forecast_df = recursive_forecast(model, df['PJME_MW'], FEATURES, n_hours=hours_to_predict)
            target_forecast_df = full_forecast_df[full_forecast_df.index.date == target_date]

            st.subheader(f"Future Results for {target_date}:")
            
            # عرض الإحصائيات جنب بعض
            col1, col2 = st.columns(2)
            with col1:
                avg_load = target_forecast_df['Predicted_MW'].mean()
                st.metric("Avg Predicted Load (MW)", f"{avg_load:,.2f}")
            with col2:
                max_load = target_forecast_df['Predicted_MW'].max()
                st.metric("Max Predicted Peak (MW)", f"{max_load:,.2f}")
                
            st.write("---")
            # عرض الجدول بعرض الشاشة
            st.dataframe(target_forecast_df.style.format("{:.2f}"), use_container_width=True)