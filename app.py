import streamlit as st
import pandas as pd
from models import (
    run_regression_models,
    run_classification_models,
    run_clustering_models,
    predict_cost,
    get_best_regression_model,
    run_what_if_analysis,
    run_profitability_calculator
)
from eda import perform_eda

st.set_page_config(layout="wide", page_title="Crop Cost Analysis Dashboard")


@st.cache_data
def load_data(path='data/cost-of-cultivation.csv'):
    df = pd.read_csv(path)
    # Basic cleaning
    df['year'] = df['year'].str.split('-').str[0].astype(int)
    # Feature Engineering
    df['total_human_labor_cost'] = df[
        ['opr_cost_hmn_lab_family', 'opr_cost_hmn_lab_attached', 'opr_cost_hmn_lab_casual']].sum(axis=1)
    df['total_machine_labor_cost'] = df[['opr_cost_mch_lab_hired', 'opr_cost_mch_lab_owned']].sum(axis=1)
    if 'net_return' not in df.columns:
        df['net_return'] = (df['main_product_value'] + df['by_product_value']) - df['cul_cost_c2']
    return df


# --- Main App ---
st.title("🌾 Indian Crop Cost & Profitability Analysis Dashboard")
st.markdown("""
This interactive dashboard provides a comprehensive analysis of crop cultivation costs in India. 
Navigate through the pages using the sidebar to explore the data, evaluate predictive models, and run your own cost simulations.
""")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Error: 'data/cost-of-cultivation.csv' not found. Please make sure the CSV file is in a 'data' subdirectory in your project folder.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Exploratory Data Analysis",
    "📈 Model Performance Evaluation",
    "🔮 Predict Future Cost",
    "🔬 What-If Scenario Analysis",
    "💰 Profitability Calculator"
])

# --- Page Content ---
if page == "🏠 Home":
    st.header("Welcome to the Dashboard!")
    st.markdown(
        "This tool is designed for agricultural analysts, policymakers, and farmers to gain insights from the Cost of Cultivation dataset.")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.markdown(f"The dataset contains **{df.shape[0]}** records and **{df.shape[1]}** columns.")

elif page == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.markdown(
        "This section provides a visual overview of the dataset, based on the key findings from the EDA report.")
    perform_eda(df)

elif page == "📈 Model Performance Evaluation":
    st.header("📈 Model Performance Evaluation")
    st.markdown(
        "This page provides a deep dive into the performance of various machine learning models. Use the tabs below to select a model type and then choose a specific model to evaluate.")

    tab1, tab2, tab3 = st.tabs(["Regression Models", "Classification Models", "Clustering Models"])

    with tab1:
        st.subheader("1. Regression Model Evaluation")
        reg_model_choice = st.selectbox(
            "Select a Regression Model to Evaluate:",
            ["XGBoost Regressor", "Random Forest Regressor", "Support Vector Regressor (SVR)",
             "Multiple Linear Regression"],
            key="reg_select"
        )
        run_regression_models(df, model_name=reg_model_choice)

    with tab2:
        st.subheader("2. Classification Model Evaluation")
        class_model_choice = st.selectbox(
            "Select a Classification Model to Evaluate:",
            ["XGBoost Classifier", "Random Forest Classifier", "Support Vector Classifier (SVC)"],
            key="class_select"
        )
        run_classification_models(df, model_name=class_model_choice)

    with tab3:
        st.subheader("3. Clustering Model Evaluation")
        cluster_model_choice = st.selectbox(
            "Select a Clustering Model to Evaluate:",
            ["K-Means", "DBSCAN", "Agglomerative Clustering"],
            key="cluster_select"
        )
        run_clustering_models(df, model_name=cluster_model_choice)


elif page == "🔮 Predict Future Cost":
    st.header("🔮 Predict Future Crop Cultivation Cost")
    st.markdown(
        "Select a state, crop, and future year to predict the total cost of cultivation (`cul_cost_c2`) per hectare.")

    # Find the best model based on performance
    best_model = get_best_regression_model(df)
    st.success(
        f"**Recommended Model:** Based on overall performance (highest R-squared), the **{best_model}** is the recommended choice for this dataset.")

    model_options = ["XGBoost Regressor", "Random Forest Regressor", "Support Vector Regressor (SVR)",
                     "Multiple Linear Regression"]
    default_index = model_options.index(best_model) if best_model in model_options else 0

    model_choice = st.selectbox(
        "Select a Model for Prediction:",
        model_options,
        index=default_index
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("Select State", sorted(df['state_name'].unique()))
    with col2:
        crop = st.selectbox("Select Crop", sorted(df['crop_name'].unique()))
    with col3:
        current_year = pd.to_datetime('today').year
        future_year = st.number_input("Select Year for Prediction", min_value=current_year, max_value=current_year + 10,
                                      value=current_year + 1)

    if st.button("Predict Cost"):
        input_data = {'state_name': state, 'crop_name': crop, 'year': future_year}
        prediction, model = predict_cost(df, input_data, model_name=model_choice)

        if prediction is not None:
            st.success(f"**Predicted Cultivation Cost:**")
            st.metric(label=f"For {crop} in {state} in {future_year}", value=f"₹ {prediction:,.2f} per Hectare")
        else:
            st.error(f"Prediction failed. There is no historical data for **{crop}** in **{state}** in the dataset.")

elif page == "🔬 What-If Scenario Analysis":
    st.header("🔬 What-If Scenario Analysis")
    run_what_if_analysis(df)

elif page == "💰 Profitability Calculator":
    st.header("💰 Profitability Calculator")
    run_profitability_calculator(df)

