Order Cancellation Prediction

A machine learning web application that predicts order cancellation probability and risk level using XGBoost and Streamlit.

Features

- Single-order cancellation prediction
- Bulk prediction using CSV and Excel files
- Cancellation probability and risk classification
- Key factors influencing predictions

Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn, XGBoost
- Joblib
- Docker

Run Locally

pip install -r requirements.txt
streamlit run deploy.py

Docker

docker build -t order-cancellation-app .
docker run -p 8501:8501 order-cancellation-app

Open "http://localhost:8501" in your browser.

Direct app link:
https://order-cancellation-pred.streamlit.app/

Model

The application uses a trained XGBoost model for predictions.

Author

Shashank Shukla
