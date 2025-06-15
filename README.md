# 🛡️ Fraud Detection in Job Postings

## 📋 Overview

This project implements machine learning algorithms to detect **fraudulent job postings** and protect job seekers from employment scams. The system analyzes various features of job listings to identify and flag potentially fake or suspicious opportunities.

---

## 🎯 Problem Statement

With the rise of online job platforms, fraudulent job postings have become increasingly common. These fake listings can lead to:

- 🕵️ Identity theft  
- 💸 Financial scams  
- ⏳ Wasted time for genuine job seekers  
- 🏢 Damage to legitimate companies' reputations  

👉 This project aims to **automatically identify and flag suspicious job postings using machine learning techniques**.

---

## ✨ Features

- ⚙️ **Data Preprocessing**: Clean and prepare job posting data for analysis  
- 🧠 **Feature Engineering**: Extract meaningful features from job descriptions and company information  
- 🤖 **Multiple ML Models**: Compare various classification algorithms  
- 📈 **Model Evaluation**: Comprehensive performance metrics  
- 🚀 **Prediction Pipeline**: Real-time fraud detection for new job postings  
- 📊 **Visualization**: Data insights + model performance charts  

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** — Data manipulation and analysis  
- **NumPy** — Numerical computing  
- **Scikit-learn** — Machine learning algorithms  
- **Matplotlib / Seaborn** — Data visualization  
- **NLTK / spaCy** — Natural language processing  
- **Jupyter Notebook / Streamlit** — Development & UI  

---

## 📊 Dataset

The dataset typically includes:  

- Job title and description  
- Company information  
- Salary details  
- Location  
- Required qualifications  
- Contact information  
- Fraud labels (genuine / fraudulent)

---

## 🧠 Machine Learning Models

The following algorithms were implemented and compared:

| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|---------:|----------:|-------:|---------:|
| Random Forest          | 94.2%    | 89.1%     | 87.3%  | 88.2%    |
| Support Vector Machine | 92.8%    | 85.4%     | 88.9%  | 87.1%    |
| Logistic Regression    | 91.5%    | 83.2%     | 89.7%  | 86.3%    |

Other models explored:
- Gradient Boosting  
- Neural Networks  
- Naive Bayes  

---

## ⚙️ How it Works

1️⃣ User enters job details or uploads a dataset  
2️⃣ App preprocesses text + engineered features  
3️⃣ ML model predicts fraud probability  
4️⃣ Results displayed with risk score, fraud class, charts, and downloadable reports  

---

## 🔍 Key Fraud Detection Signals

### ✉️ Textual Analysis  
- Poor grammar / spelling  
- Unrealistic salary offers  
- Vague descriptions  
- Urgent hiring language  

### 🏢 Company Information  
- Unverified company name  
- Missing contact info  
- Suspicious email domains  

### 📌 Job Requirements  
- Minimal/no qualifications required  
- Excessive work-from-home emphasis  
- Upfront payment requests  

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness  
- **Precision**: How many predicted frauds are true frauds  
- **Recall**: How many actual frauds are detected  
- **F1-Score**: Balance of precision and recall  
- **ROC-AUC**: Overall model discrimination ability  

---

🚀 How to Run This Project

1️⃣ Clone this repo
bash
Copy
Edit
git clone https://github.com/yourusername/your-repo.git
cd your-repo

2️⃣ Set up Python environment
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

3️⃣ Run the Streamlit app
bash
Copy
Edit
streamlit run app/app.py
Then visit: http://localhost:8501 in your browser.

## Video Link 

Link:  https://drive.google.com/drive/folders/1GgJfhA1iR8iwQ1QAzllVIOa2_n9Uh9-e?usp=drive_link
