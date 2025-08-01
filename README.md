# 🏨 MLOps Hotel Revenue Management System

A Predictive Analytics System for Hotel Reservation Status using MLOps Best Practices



## 📖 Project Description

The MLOps-Hotel-Revenue-Management-System is a machine learning application that predicts hotel booking cancellation status based on historical reservation data. This system aids revenue managers and hospitality stakeholders in improving booking forecasts, optimizing overbooking strategies, and enhancing customer retention by identifying potential cancellations before they occur.

This project simulates a real-world hotel revenue management workflow by incorporating:

- End-to-end MLOps pipeline for data ingestion, processing, model training, evaluation, and deployment
- A LightGBM classification model optimized for performance on imbalanced booking data
- MLFlow for experiment tracking and model registry
- A Flask-based frontend for real-time inference, allowing hotel managers to input reservation details and receive booking status predictions

## 🚀 Features

### Data Pipeline
- Automated data ingestion and preprocessing
- Label encoding and skewness handling
- Feature selection and engineering

### ML Model
- LightGBM classifier with hyperparameter tuning
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Handling of imbalanced booking data

### MLOps Integration
- MLFlow for experiment tracking
- Metric logging and model registry
- Model versioning and deployment

### Web Interface
- User-friendly Flask frontend
- Real-time prediction capability
- Form for inputting reservation details

## 🧩 Use Case: Hotel Revenue Management

In the hospitality industry, booking cancellations significantly impact revenue forecasting and operational efficiency. This system empowers revenue managers to:

- Predict cancellations preemptively based on key booking attributes (lead time, room type, market segment, etc.)
- Implement proactive retention strategies (e.g., targeted offers, confirmations) for bookings likely to be canceled
- Optimize room allocation and overbooking policies to maximize occupancy rates while minimizing revenue loss
- Integrate with existing hotel management systems through APIs for scalable deployment

## ⚙️ Tech Stack

| Component          | Tool/Technology       |
|--------------------|-----------------------|
| Data Processing    | Pandas, NumPy         |
| Modeling           | LightGBM (Classification) |
| MLOps & Tracking   | MLFlow                |
| Web Framework      | Flask, FastAPI               |
| Frontend           | HTML, CSS             |

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VibhavAhuja19/Hotel-Reservation.git
cd Hotel-Rservation
```

## 2️⃣ Set Up the Environment
```bash
python3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 3️⃣ Run the Flask and FastAPI Application
```bash
For Flask - python app.py
For FastAPI - uvicorn main:app --reload
``` 

