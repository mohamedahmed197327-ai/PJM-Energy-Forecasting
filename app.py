import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="PJME Load Forecasting", layout="wide")
st.title("Smart Grid Load Forecasting (PJME)")

# 1. تحديث قائمة الميزات لتطابق الموديل الجديد
FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']

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

try:
    model = load_model()
    df = load_data()
    st.success("Model and data loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or data. Details: {e}")
    st.stop()

last_available_date = df.index.max().date()
first_available_date = df.index.min().date()

st.sidebar.header("⚙️ Forecast Settings")
target_date = st.sidebar.date_input(
    "Select Target Date:",
    value=last_available_date + pd.Timedelta(days=1),
    min_value=first_available_date, 
    max_value=None
)

def create_features(date_index):
    """دالة بسيطة لاستخراج خصائص الوقت مباشرة"""
    X = pd.DataFrame(index=date_index)
    X['hour'] = X.index.hour
    X['dayofweek'] = X.index.dayofweek
    X['quarter'] = X.index.quarter
    X['month'] = X.index.month
    X['year'] = X.index.year
    X['dayofyear'] = X.index.dayofyear
    X['dayofmonth'] = X.index.day
    X['weekofyear'] = X.index.isocalendar().week.astype(int)
    return X[FEATURES]

if st.sidebar.button("Get Forecast "):
    with st.spinner(f'Processing data for {target_date}...'):
        
        # إنشاء الساعات لليوم المطلوب (سواء ماضي أو مستقبل)
        target_hours = pd.date_range(start=f"{target_date} 00:00:00", end=f"{target_date} 23:00:00", freq='h')
        X_predict = create_features(target_hours)
        predictions = model.predict(X_predict)
        
        results_df = pd.DataFrame(index=target_hours)
        results_df['Predicted_MW'] = predictions
        
        st.subheader(f"Results for {target_date}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Predicted Load (MW)", f"{predictions.mean():,.2f}")
        with col2:
            st.metric("Max Predicted Peak (MW)", f"{predictions.max():,.2f}")
            
        st.write("---")
        
        # إضافة القيم الحقيقية إذا كان التاريخ في الماضي للتقييم
        if target_date <= last_available_date:
            actual_data = df.loc[df.index.date == target_date, 'PJME_MW']
            results_df['Actual_MW'] = actual_data
            
        st.dataframe(results_df.style.format("{:.2f}"), use_container_width=True)