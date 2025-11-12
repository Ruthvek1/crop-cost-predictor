# 📊 Project Report: Indian Crop Cost & Profitability Analysis Dashboard

**Student Name:** Ruthvek Kannan  
**PRN:** 22070521031  
**Institute:** Symbiosis Institute of Technology, Nagpur Campus (SIT-N)  
**Department:** Computer Science and Engineering  
**Mentor:** Dr. Piyush Chauhan  
**Session:** 2025-26  

---

## 🧠 Abstract

The Indian agricultural sector, pivotal to the national economy, faces critical challenges regarding the rising and volatile costs of cultivation. Accurate prediction and analysis of these costs are essential for ensuring the financial stability of farmers and informing policy decisions.

This project addresses this need by developing an end-to-end data science solution: **the Indian Crop Cost & Profitability Analysis Dashboard.**

This report documents the complete lifecycle of the project, from data ingestion to the deployment of a multi-functional web application. We utilized the *“Cost of Cultivation of Principal Crops in India”* dataset to perform a rigorous **Exploratory Data Analysis (EDA)**, revealing significant trends such as the *“cost-price squeeze”* affecting farmer profitability.

The core of the project involved training and evaluating a diverse suite of machine learning models across three domains:

- **Regression:** To predict exact production costs per quintal.  
- **Classification:** To categorize farms into cost-efficiency tiers.  
- **Clustering:** To discover unsupervised “farm archetypes” based on operational data.  

Using a robust 5-fold cross-validation strategy, we identified **XGBoost Regressor** as the optimal model for cost prediction and **Support Vector Classifier (SVC)** for efficiency classification. These models were then integrated into an interactive **Streamlit dashboard**.

The final application empowers users with advanced tools, including a **Future Cost Predictor**, a **What-If Scenario Analyzer**, and a **Profitability Calculator**, bridging the gap between complex data science and actionable agricultural insights.

---

## 📚 Table of Contents

1. [Introduction](#1-introduction)  
2. [System Design and Methodology](#2-system-design-and-methodology)  
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)  
4. [Modeling and Evaluation](#4-modeling-and-evaluation)  
5. [Application Development (Streamlit)](#5-application-development-streamlit)  
6. [Conclusion and Future Scope](#6-conclusion-and-future-scope)  
7. [References](#7-references)

---

## 1. Introduction

### 1.1 Project Background
Agriculture employs a vast segment of India’s population. However, the economic viability of farming is increasingly threatened by the rising costs of inputs — labor, machinery, fertilizers, and seeds. The *“Comprehensive Scheme for studying Cost of Cultivation of Principal Crops in India,”* managed by the Ministry of Agriculture, collects granular data on these costs. However, this data often remains inaccessible to the average stakeholder in its raw format.

### 1.2 Problem Statement
Stakeholders in the Indian agricultural ecosystem lack accessible, data-driven tools to analyze historical cost trends, predict future expenses, and simulate the financial impact of market fluctuations. This information asymmetry hinders effective planning, budgeting, and policy formulation.

### 1.3 Objectives
The primary objective is to **democratize access** to agricultural cost insights through a user-friendly dashboard.

- **Analyze:** Visualize historical trends in cost, yield, and net returns across different states and crops.  
- **Model:** Build robust ML models capable of predicting future costs and classifying operational efficiency.  
- **Simulate:** Provide interactive tools that allow users to perform *“what-if”* analyses on input costs and estimate profitability.

---

## 2. System Design and Methodology

### 2.1 Technology Stack

The project was implemented using a Python-based stack:

- **Streamlit:** For rapid web application development and deployment.  
- **Pandas & NumPy:** For efficient data manipulation and numerical operations.  
- **Scikit-learn:** For preprocessing pipelines and ML algorithms (Linear Regression, Random Forest, SVM, K-Means).  
- **XGBoost:** For high-performance gradient boosting.  
- **Matplotlib & Seaborn:** For static data visualization.  
- **Plotly:** For interactive charts in the dashboard.

### 2.2 System Architecture

The application follows a modular architecture.  
The `app.py` script serves as the controller, handling navigation and user input, while specialized modules (`models.py`, `eda.py`, `advanced_features.py`) manage data processing and modeling.

**Flowchart:**
<p align="center">
<img width="500" height="700" alt="image" src="https://github.com/user-attachments/assets/4acd20a0-a5fa-4b89-89d3-f0ef9e7b0548" />

*Caption: Fig 2. Data Preprocessing and Modeling Pipeline*
</p>

## 3. Exploratory Data Analysis (EDA)

### 3.1 Data Distribution and Structure

The dataset includes multiple crop categories, with **Cereals** and **Pulses** being most represented.  
Major agricultural states such as **Maharashtra** and **Uttar Pradesh** contribute significantly.

**Key Finding:**  
While cultivation costs have risen linearly, **net returns** have remained volatile and stagnant.
<p align="center">
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/4559d6b1-428a-4e63-85f3-32f8a8d855eb" />
</p>
*Caption: Fig 1.1. Trend of Costs and Returns Over Time*

### 3.2 Multivariate Analysis

**Correlation Analysis:**  
`cul_cost_c2 (Total Cost)` shows a strong correlation with variables like `total_human_labor_cost` and `opr_cost_fertilizer`, validating feature selection.
<p align="center">
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/e07d9724-74f0-48e8-ab73-74df063c3d21" />
</p>

*Caption: Fig 1.2. Correlation Matrix of Key Variables*

---

## 4. Modeling and Evaluation

A **5-Fold Cross-Validation** strategy ensured robust and unbiased performance metrics.

### 4.1 Regression Analysis (Cost Prediction)

**Goal:** Predict `prod_cost_c2`

| Model | R² Score | RMSE | Key Notes |
|-------|-----------|------|------------|
| Multiple Linear Regression | -0.072 | – | Relationship is non-linear. Residuals show heteroscedasticity. |
| Random Forest Regressor | 0.881 | – | Captured non-linearities effectively. |
| **XGBoost Regressor (Best)** | **0.899** | **703.33** | Highest accuracy, lowest error. |

*Caption: Fig 4.1. Actual vs Predicted Plot for XGBoost Regressor*

### 4.2 Classification Analysis (Efficiency)

**Goal:** Predict `Cost_Efficiency_Class (Lowest, Below Avg, Above Avg, Highest)`

| Model | Accuracy | MCC | Observation |
|--------|-----------|------|-------------|
| XGBoost Classifier | 53.14% | – | Struggled with multi-class boundaries. |
| **Support Vector Classifier (Best)** | **70.9%** | **0.612** | Robust and stable performance. |

*Caption: Fig 8.1. ROC Curves for SVC*

### 4.3 Clustering Analysis (Farm Archetypes)

**Goal:** Discover unsupervised clusters using K-Means and DBSCAN.

| Model | Score | Insights |
|--------|--------|-----------|
| K-Means (K=4) | Silhouette = 0.348 | Revealed distinct “farm archetypes.” |
| DBSCAN | – | Useful for anomaly detection (outliers). |

*Caption: Fig 11.1. DBSCAN Clustering and Outlier Detection*

---

## 5. Application Development (Streamlit)

### 5.1 Predictive Tools

The **“Predict Future Cost”** page integrates the **XGBoost Regressor**, enabling future cost forecasts (e.g., for 2026).

*Caption: Fig 13.1. Future Cost Prediction Interface*

### 5.2 Advanced Analytical Features

#### 🧩 “What-If” Scenario Analysis
Allows users to modify input parameters (e.g., fertilizer prices) via sliders to instantly recompute total costs.

*Caption: Fig 13.2. What-If Scenario Analysis Tool*

#### 💰 Profitability Calculator
Combines cost prediction with expected yield and market price to calculate:

- Projected Revenue  
- Net Profit  
- Break-even Price  

*Caption: Fig 13.3. Profitability Calculator and Financial Outlook*

---

## 6. Conclusion and Future Scope

### ✅ Conclusion
The project demonstrates how **machine learning can effectively model agricultural economics.**  
By deploying **XGBoost** and **SVC** through an intuitive **Streamlit dashboard**, we provide actionable insights for both farmers and policymakers.

### 🔮 Future Scope

- Integration of **real-time API data** for market prices (Mandis).  
- Incorporation of **weather and soil datasets** for precision modeling.  
- Deployment to **public cloud** for greater accessibility.

---

## 7. References

1. Directorate of Economics and Statistics, Ministry of Agriculture.  
   *“Comprehensive Scheme for studying Cost of Cultivation of Principal Crops in India.”*  
2. [Scikit-learn Documentation](https://scikit-learn.org/)  
3. [Streamlit Documentation](https://docs.streamlit.io/)

---
