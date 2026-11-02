<img width="245" height="2268" alt="image" src="https://github.com/user-attachments/assets/e169ebea-a459-4333-8678-3ddfa140ff05" />
A Machine Learning-based web application that predicts customer churn in the telecom industry using advanced data preprocessing, feature engineering, predictive modeling, and interactive analytics.

🚀 Project Overview
Customer churn is a major challenge for telecom companies. This project analyzes customer behavior and predicts whether a customer is likely to leave the service.

The solution combines:

📊 Exploratory Data Analysis (EDA)
⚙️ Feature Engineering
🤖 Machine Learning Models
📈 Interactive Analytics
🌐 Streamlit Web Application
🎯 Objectives
Identify customers at risk of churn
Analyze factors influencing customer retention
Compare multiple machine learning models
Build an interactive prediction system
Support data-driven business decisions
📂 Dataset Features
The dataset contains customer information including:

Gender
Senior Citizen
Partner
Dependents
Tenure
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Contract Type
Paperless Billing
Payment Method
Monthly Charges
Total Charges
Churn
🔧 Project Workflow
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Encoding & Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Streamlit Deployment
🧠 Machine Learning Models Used
1. Logistic Regression
Used as a baseline classification model.

2. Decision Tree Classifier
Used to capture nonlinear customer behavior patterns.

3. Random Forest Classifier ✅
Selected as the final model due to superior performance.

📊 Model Performance
Model	Accuracy
Logistic Regression	26.54%
Decision Tree	48.62%
Random Forest	73.10%
✅ Best Performing Model: Random Forest Classifier

📈 Feature Engineering
The following preprocessing and feature engineering techniques were applied:

Missing Value Handling
Label Encoding
One-Hot Encoding
Feature Scaling
Customer Behavior Analysis
Feature Selection
Final Features:

SeniorCitizen
tenure
MonthlyCharges
TotalCharges
gender_Male
Partner_Yes
Dependents_Yes
PhoneService_Yes
MultipleLines_No phone service
MultipleLines_Yes
InternetService_Fiber optic
InternetService_No
OnlineSecurity_No internet service
OnlineSecurity_Yes
OnlineBackup_No internet service
OnlineBackup_Yes
DeviceProtection_No internet service
DeviceProtection_Yes
TechSupport_No internet service
TechSupport_Yes
StreamingTV_No internet service
StreamingTV_Yes
StreamingMovies_No internet service
StreamingMovies_Yes
Contract_One year
Contract_Two year
PaperlessBilling_Yes
PaymentMethod_Credit card (automatic)
PaymentMethod_Electronic check
PaymentMethod_Mailed check
🖥️ Streamlit Application Features
🏠 Dashboard
Total Customers
Churned Customers
Retained Customers
Model Accuracy
🤖 Churn Prediction
Customer Information Form
Real-Time Prediction
Churn Probability Score
Risk Classification
📊 Analytics
Customer Insights
Churn Analysis
Business Intelligence Visualizations
⭐ Feature Importance
Key Churn Drivers
Business Insights
📈 Model Comparison
Logistic Regression
Decision Tree
Random Forest
🛠️ Technologies Used
Programming
Python
Data Analysis
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Plotly
Machine Learning
Scikit-Learn
Deployment
Streamlit
Version Control
Git
GitHub
📦 Installation
Clone the repository:

git clone https://github.com/YOUR_USERNAME/telecom-customer-churn-prediction.git
Move into project folder:

cd telecom-customer-churn-prediction
Install dependencies:

pip install -r requirements.txt
Run the application:

python -m streamlit run app.py
📁 Project Structure
telecom-customer-churn-prediction/
│
├── app.py
├── Telecom.ipynb
├── churn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── feature_importance.csv
├── requirements.txt
├── README.md
└── dataset.csv
📸 Dashboard Preview
Customer Churn Analytics Dashboard
Add your dashboard screenshot here

Churn Prediction Dashboard
Add your prediction dashboard screenshot here

💡 Key Business Insights
Month-to-month contract customers show higher churn risk.
Customers with fiber optic internet services are more likely to churn.
Customers with shorter tenure exhibit higher churn probability.
Long-term customers demonstrate stronger retention behavior.
🔮 Future Enhancements
Power BI Integration
Customer Segmentation
XGBoost Implementation
Customer Retention Recommendation Engine
Cloud Deployment
Real-Time API Integration
👩‍💻 Author
Sakshi

Aspiring Data Analyst | Data Scientist | Machine Learning Enthusiast

⭐ Support
If you found this project helpful, please consider giving it a ⭐ on GitHub.

📜 License
This project is developed for educational and portfolio purposes
