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
