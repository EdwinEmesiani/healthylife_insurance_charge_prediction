> Part of my **deployable AI systems portfolio**. See my [GitHub profile](https://github.com/EdwinEmesiani) for related projects like **Technoecom E-Commerce EDA** and **RAG_10k**.


# HealthyLife Insurance Charge Prediction – Deployable ML System

> As an engineer building deployable AI systems, this project turns a **regression model** into a **live insurance pricing service** with monitoring and drift awareness.

---

## 1. Project Overview

HealthyLife Insurance (fictional) is a leading insurer headquartered in New York, USA, serving customers with **health, auto, and life** products.

This repository implements an **end-to-end regression system** to predict **health insurance charges** for prospective customers based on:

- Age  
- Sex  
- BMI  
- Number of children  
- Smoker status  
- Region  

The project goes beyond a notebook model and demonstrates:

- **Exploratory analysis** of customer risk patterns  
- **Production-style preprocessing and training pipeline**  
- **Deployed Gradio application** on Hugging Face Spaces  
- **Logging and monitoring** concepts for data and model drift  
- A foundation for **MLOps workflows** (retraining & CI/CD).

---

## 2. System Architecture (Text Diagram)

```text
Offline
-------
insurance.csv
      ↓
pandas DataFrame
      ↓
EDA (distributions, relationships)
      ↓
Train/Test Split
      ↓
Preprocessing Pipeline (ColumnTransformer):
    - StandardScaler(age, bmi, children)
    - OneHotEncoder(sex, smoker, region)
      ↓
Models:
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor (selected)
      ↓
Model Evaluation (MSE, R²)
      ↓
Serialized Pipeline (model.joblib)


Online (Hugging Face Space)
---------------------------
Client (Gradio UI / API)
      ↓
app.py
      ↓
Load model.joblib
      ↓
Validate & Transform Inputs (inside pipeline)
      ↓
Predict insurance charges
      ↓
Log inputs + predictions (CommitScheduler → HF Dataset)
      ↓
Monitoring & Drift Analysis (notebook / dashboards)

3. Key Features
EDA layer:

Distribution of charges by sex and smoking status.

Age and BMI trends vs charges.

Clear visualisation of risk factors (especially smoking and older age).

Three regression models compared:

LinearRegression

DecisionTreeRegressor

RandomForestRegressor (best performer: highest R², lowest MSE).

Robust preprocessing with scikit-learn Pipelines:

ColumnTransformer with StandardScaler and OneHotEncoder.

Ensures training and inference use the same transformations.

Model packaging:

train.py encapsulates data loading, training, evaluation, and model.joblib creation.

requirements.txt declares all dependencies.

Deployment:

app.py provides a Gradio UI for HealthyLife underwriters and non-technical stakeholders.

Hosted on Hugging Face Spaces.

Monitoring & drift concept:

Requests and predictions logged (e.g. to logs/ and a Hub dataset via CommitScheduler).

Notebook logic to compare training target distribution vs live prediction distribution and flag potential model drift.

Simple checks for data drift on key features (e.g. age, BMI).

4. Tech Stack
Language: Python

ML & Data: pandas, numpy, scikit-learn, joblib

Serving: gradio, Hugging Face Spaces

Monitoring / Logging: huggingface_hub.CommitScheduler, JSON logs (with drift analysis in notebook)

5. How to Run Locally
5.1. Train the Model
git clone https://github.com/<your-username>/healthylife_insurance_charge_prediction.git
cd healthylife_insurance_charge_prediction

pip install -r requirements.txt

python train.py
# -> creates model.joblib
5.2. Run the Gradio App
python app.py
Then open the local URL printed by Gradio (usually http://127.0.0.1:7860).


6. Example Business Insight
•	Smokers and older customers with high BMI have disproportionately higher charges, which confirms risk-based pricing patterns.
•	The Random Forest model captures these non-linear interactions better than a simple linear model, leading to more accurate charge estimation.
________________________________________
7. Monitoring & Data Drift (Conceptual)
The project sketches a monitoring loop:
•	Logging: Each prediction logs features + predicted charges.
•	Model drift: Compare the distribution of training targets vs live predictions. Significant shifts can trigger retraining.
•	Data drift: Compare feature means/variances (e.g. age, BMI) between training data and recent live traffic.
These checks can be extended with EvidentlyAI dashboards and integrated into a CI/CD pipeline with GitHub Actions.
________________________________________
8. Repository Structure
.
├── data/
│   └── insurance.csv
├── notebooks/
│   └── Edwin_Emesiani_Ejiofor_insurance_charge_prediction.ipynb
├── app.py
├── train.py
├── model.joblib
├── requirements.txt
└── README.md
________________________________________
9. Possible Next Steps
•	Add proper monitoring dashboards (EvidentlyAI) and alerting.
•	Implement a scheduled retraining job (e.g. GitHub Actions + HF Hub).
•	Extend the API surface: batch inferencing endpoints, multi-policy simulations, or scenario testing for pricing teams.





