# 🎫 Customer Support Ticket Classifier

***This project classifies customer support tickets into categories such as:***
- Technical issue
- Billing inquiry
- Product inquiry
- Cancellation request
- Refund request

***The project includes:***
- Data cleaning
- Exploratory data analysis (EDA)
- Model training
- API using FastAPI
- Frontend UI using Streamlit
- Saving model and vectorizer for production

### ---This README documents the exact steps, files, and pipeline used in your project---

## 📂 Project Structure

```
customer-support-classifier/
│
├── api/
│   └── main.py                      # FastAPI backend (Model API)
│
├── app/
|   └── app.py                       # Streamlit UI
│   └── assets/
│       └── style.css                # Custom UI styling
|   └──pages/
|         └── API_Tester.py 
│         └── EDA.py
|
├── data/
│   └── clean_tickets.csv           # Stored after preprocessing
│
├── models/
│   ├── model.pkl                   # Trained ML model
│   ├── vectorizer.pkl              # TF-IDF vectorizer
│
├── notebooks/
│   └── EDA.ipynb                   # EDA + cleaning notebook
│   └──customer_support_tickets.csv
|   └── data/
|   |    └── clean_tickets.csv
|   └── test.py
| 
|
├── src/
│   ├── preprocess.py               # Text cleaning & normalization
│   ├── predict.py                  # predictions
│   └── train.py                    # Model training script
│
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```



## 📊Workflow Overview
Performed inside notebooks/EDA.ipynb.

***Main cleaning actions you used:***

- Selecting the important columns:
  - Ticket Subject
  - Ticket Description
  - Ticket Type
  - Product Purchased

- Lowercasing
- Stripping extra spaces
- Normalizing inconsistent labels
- Converting refund/billing/payment → billing inquiry
- Converting tech/bug/error/crash → technical issue
- Removing "other" class
- Fixing mislabeled rows using keywords
- Dropping ticket descriptions shorter than 5 characters
- Saving cleaned file:
  - data/cleaned_tickets.csv
### Purpose: Clean, consistent text → better model performance.**



## 📉EDA (Exploratory Data Analysis)

**Missing values heatmap**
(Purpose: See which columns have missing data.)

**Ticket Type distribution**
(Purpose: Check class balance before training.)

**Priority distribution**
(Purpose: Understand metadata distribution.)



## 🧪Feature Engineering

***Inside train.py:***
- Combine Subject + Description
- TF-IDF vectorizer with n-grams
- Label encoding
- Train-test split



## 𝌭Model Training

***Inside src/train.py:***
- Logistic Regression (with class balancing)
- RandomForest fallback
- Save:
    - models/model.pkl
    - models/vectorizer.pkl
    - models/label_encoder.pkl



## 🔥FastAPI Backend

***Runs from:***
- api/main.py

***Endpoint:***
- POST /predict
  
***Sends JSON → returns:***
- ticket_type
- confidence

***Run FastAPI:***
- uvicorn api.main:app --reload


## ֎Streamlit Frontend

***Runs from:***
- app/app.py

***Features:***
- Input fields: subject, description, product purchased
- Sends request to FastAPI
- Shows prediction
- Displays confidence
- Shows model accuracy

***Run Streamlit:***
- streamlit run app/app.py


## 🛠Installation Steps

***Installing requirements***
- python -m venv venv
- source venv/bin/activate     # mac
- pip install -r requirements.txt


***Train model:***
- python src/train.py
- Start API:
- uvicorn api.main:app --reload


***Start Streamlit:***
- streamlit run app/app.py

## 🎯Example Prediction

***Input:***
- App not turning on after update.

***Output:***
- technical issue

### App Screenshots

**Ticket Prediction**

![App Dashboard](App_Dashboard.png)



