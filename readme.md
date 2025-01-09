# 🚗 GetAround Pricing Optimization Project

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-blue?style=for-the-badge&logo=mlflow&logoColor=white)

## 📊 Project Overview
This project implements a machine learning solution for optimizing rental car pricing for GetAround. It includes data analysis, model training, API deployment, and a Streamlit dashboard for price predictions.

## 🗂️ Repository Structure
```
├── notebooks/
│   ├── EDA.ipynb
│   └── modeling.ipynb
├── src/
│   ├── get_around_pricing_project.csv
│   └── project_description.ipynb
├── streamlit/
│   └── app.py
├── api/
│   └── main.py
├── docker-compose.txt
├── Dockerfile.fastapi
├── Dockerfile.streamlit
├── Dockerfile.mlflow
├── requirements.fastapi.txt
├── requirements.mlflow.txt
├── requirements.streamlit.txt
├── train_model.py
├── .env
├── .secrets
└── README.md
```

## ⚙️ Prerequisites
- Python 3.9+
- Docker
- AWS Account with S3 access
- NeonDB Account

## 🚀 Setup Instructions

### 1. Database Setup
1. Create a NeonDB account
2. Create a new database
3. Save your database connection string

### 2. AWS Setup
1. Create an AWS account
2. Create an S3 bucket
3. Create IAM credentials (Access Key & Secret)
4. Save your AWS credentials

### 3. Environment Variables
Create a `.env` file with:
```
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your_bucket_name

# NeonDB Configuration
NEON_DATABASE_URL=your_database_url

# MLflow Configuration
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://your_bucket/
```

### 4. Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Build and run Docker container
docker-compose up --build
```

## 🛠️ Components

### 1. 📈 Data Analysis & Modeling
- `EDA.ipynb`: Exploratory Data Analysis
- `modeling.ipynb`: Model Development and Training
- `model_training.py`: Production Training Script

### 2. 🔍 MLflow Tracking
- Metrics stored in NeonDB
- Models stored in S3
- Access MLflow UI locally at `http://localhost:5000`

### 3. 🚀 FastAPI Application
- Endpoint: `/predict` for price predictions
- Documentation: `/docs` for API documentation

### 4. 📊 Streamlit Dashboard
- Interactive interface for price predictions

## 📝 Usage

### Training the Model
```bash
python model_training.py
```

### Running the API
```bash
docker-compose up
```

### Using the API
Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "model_key": "Mercedes",
           "mileage": 150000,
           "engine_power": 200,
           "fuel": "diesel",
           "paint_color": "black",
           "car_type": "sedan",
           "private_parking_available": true,
           "has_gps": true,
           "has_air_conditioning": true,
           "automatic_car": true,
           "has_getaround_connect": true,
           "has_speed_regulator": true,
           "winter_tires": true
         }'
```

## 🌐 Deployment
- API deployed on [en cours]
- Dashboard available at [en cours]
