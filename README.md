# ⚡ Smart Grid Hourly Electricity Load Forecasting

An end-to-end Machine Learning web application that forecasts hourly electricity consumption using historical PJM East (PJME) energy data, powered by **XGBoost** and deployed with **Streamlit**.

🔗 **[Live Demo on Streamlit](https://pjm-energy-forecasting-hvlafrrhmsumfeh2uhsmbb.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📌 Project Overview
Accurate electricity load forecasting is critical for smart grid management, energy distribution, and preventing power outages. This project models hourly energy demand patterns (daily, weekly, and seasonal trends) using time-series feature engineering and gradient boosting.

## ✨ Key Features
- **Time-Series Feature Engineering:** Extracted temporal features (hour, day of week, quarter, month, year, lag features) to capture seasonality.
- **XGBoost Regressor:** Tuned gradient boosting model for high-accuracy hourly load prediction.
- **Interactive Dashboard:** Built with Streamlit to visualize actual vs. predicted energy demand and generate custom forecasts.

## 📊 Dataset & Modeling
- **Dataset:** PJM Hourly Energy Consumption (`PJME_hourly.csv`) — over 10 years of hourly megawatt (MW) energy usage.
- **Model:** Tuned `XGBoost Regressor` trained on temporal and lag features to capture daily, weekly, and seasonal electricity demand patterns.
- **Evaluation:** Evaluated using standard regression metrics (RMSE and MAE) on a chronological train/test split to prevent data leakage.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohamedahmed197327-ai/PJM-Energy-Forecasting.git
