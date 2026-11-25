## 📊 Dataset & Modeling
- **Dataset:** PJM Hourly Energy Consumption (`PJME_hourly.csv`) — over 10 years of hourly megawatt (MW) energy usage.
- **Model:** Tuned `XGBoost Regressor` trained on temporal and lag features to capture daily, weekly, and seasonal electricity demand patterns.
- **Evaluation:** Evaluated using standard regression metrics (RMSE and MAE) on a chronological train/test split to prevent data leakage.
