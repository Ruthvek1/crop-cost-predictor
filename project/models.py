import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    median_absolute_error, matthews_corrcoef, silhouette_score, silhouette_samples, davies_bouldin_score, \
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.ticker import FixedLocator, FixedFormatter


# --- Helper Function for Data Preprocessing ---
def preprocess_data(df, for_clustering=False):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    if 'total_human_labor_cost' not in df_clean.columns:
        df_clean['total_human_labor_cost'] = df_clean[
            ['opr_cost_hmn_lab_family', 'opr_cost_hmn_lab_attached', 'opr_cost_hmn_lab_casual']].sum(axis=1)
    if 'total_machine_labor_cost' not in df_clean.columns:
        df_clean['total_machine_labor_cost'] = df_clean[['opr_cost_mch_lab_hired', 'opr_cost_mch_lab_owned']].sum(
            axis=1)
    if for_clustering:
        df_clean['labor_cost_ratio'] = (df_clean['total_human_labor_cost'] / df_clean['cul_cost_c2']).fillna(0)
        df_clean['machine_cost_ratio'] = (df_clean['total_machine_labor_cost'] / df_clean['cul_cost_c2']).fillna(0)
        df_clean['fertilizer_cost_ratio'] = (df_clean['opr_cost_fertilizer'] / df_clean['cul_cost_c2']).fillna(0)
        df_clean['fixed_cost_ratio'] = (df_clean['fix_cost'] / df_clean['cul_cost_c2']).fillna(0)
        df_clean['mechanization_index'] = (df_clean['total_machine_labor_cost'] + 1) / (
                df_clean['total_human_labor_cost'] + 1)
        if 'net_return' not in df_clean.columns:
            df_clean['net_return'] = (df_clean['main_product_value'] + df_clean['by_product_value']) - df_clean[
                'cul_cost_c2']
        df_clean.replace([np.inf, -np.inf], 0, inplace=True)
    return df_clean


# --- Performance Summary Functions ---
@st.cache_data
def _get_regression_performance_summary(df):
    data = preprocess_data(df)
    target = 'prod_cost_c2'
    data.dropna(subset=[target], inplace=True)
    y = data[target]
    features = ['state_name', 'crop_name', 'year', 'derived_yield', 'opr_cost_seed', 'opr_cost_fertilizer',
                'total_human_labor_cost', 'total_machine_labor_cost']
    X = data[features]
    cat_features = ['state_name', 'crop_name'];
    num_features = [col for col in features if col not in cat_features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
    models = {"Multiple Linear Regression": LinearRegression(),
              "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
              "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
              "Support Vector Regressor (SVR)": SVR(C=100, gamma=0.1)}

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

        results.append({
            "Model": name,
            "CV R-squared Mean": cv_scores.mean(),
            "CV R-squared Std": cv_scores.std()
        })
    return pd.DataFrame(results)


@st.cache_data
def _get_classification_performance_summary(df):
    data = preprocess_data(df)
    labels = ['Lowest Cost', 'Below Average Cost', 'Above Average Cost', 'Highest Cost']
    data['Cost_Efficiency_Class'] = data.groupby('crop_name')['prod_cost_c2'].transform(
        lambda x: pd.qcut(x, q=4, labels=labels, duplicates='drop'))
    data.dropna(subset=['Cost_Efficiency_Class'], inplace=True)
    le = LabelEncoder();
    data['Cost_Efficiency_Class_Encoded'] = le.fit_transform(data['Cost_Efficiency_Class'])
    target = 'Cost_Efficiency_Class_Encoded';
    y = data[target]
    features = ['state_name', 'crop_name', 'year', 'derived_yield', 'opr_cost_seed', 'opr_cost_fertilizer',
                'total_human_labor_cost', 'total_machine_labor_cost']
    X = data[features]
    cat_features = ['state_name', 'crop_name'];
    num_features = [col for col in features if col not in cat_features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
    models = {"Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
              "XGBoost Classifier": xgb.XGBClassifier(objective='multi:softmax', random_state=42,
                                                      eval_metric='mlogloss'),
              "Support Vector Classifier (SVC)": SVC(probability=True, random_state=42)}

    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        results.append({
            "Model": name,
            "CV Accuracy Mean": cv_scores.mean(),
            "CV Accuracy Std": cv_scores.std()
        })
    return pd.DataFrame(results)


# --- 1. Regression Models ---
def run_regression_models(df, model_name):
    # --- Performance Dashboard ---
    st.markdown("#### Model Performance Dashboard (based on 5-Fold Cross-Validation)")
    summary_df = _get_regression_performance_summary(df)
    st.dataframe(summary_df.set_index('Model'))

    with st.expander("About Cross-Validation"):
        st.markdown("""
        **Cross-Validation (CV)** provides a more robust measure of model performance. Instead of a single train/test split, it splits the data into 5 'folds', trains on 4, and tests on 1, repeating this 5 times.
        - **CV R-squared Mean:** The average R-squared across all 5 folds. This is a more reliable estimate of the model's true performance.
        - **CV R-squared Std:** The standard deviation of the scores. A low value indicates that the model's performance is stable and consistent across different subsets of the data.
        """)

    fig_summary, ax_summary = plt.subplots()
    sns.barplot(x="CV R-squared Mean", y="Model", data=summary_df.sort_values("CV R-squared Mean", ascending=False),
                ax=ax_summary)
    ax_summary.set_title("Model Comparison by Cross-Validated R-squared")
    st.pyplot(fig_summary)
    plt.close(fig_summary)
    st.markdown("---")

    # --- Detailed Evaluation on a Single Split ---
    st.markdown("#### Detailed Evaluation on a Single Train/Test Split")
    data = preprocess_data(df)
    target = 'prod_cost_c2'
    data.dropna(subset=[target], inplace=True)
    y = data[target]
    features = ['state_name', 'crop_name', 'year', 'derived_yield', 'opr_cost_seed', 'opr_cost_fertilizer',
                'total_human_labor_cost', 'total_machine_labor_cost']
    X = data[features]

    st.info(f"**Features Used for Prediction:** `{'`, `'.join(features)}`")

    cat_features = ['state_name', 'crop_name']
    num_features = [col for col in features if col not in cat_features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "Support Vector Regressor (SVR)": SVR(C=100, gamma=0.1)  # Using better default hyperparameters
    }
    model = models[model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader(f"--- Detailed Evaluation for {model_name} ---")
    r2 = r2_score(y_test, y_pred)
    rmse, mae = np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)
    mae_perc = (mae / y_test.mean()) * 100
    medae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    if model_name == "Multiple Linear Regression":
        n = X_test.shape[0];
        p = X_train.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Adj. R-squared", f"{adj_r2:.3f}")
            st.metric("Explained Variance", f"{evs:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("Median Abs. Error", f"{medae:.2f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared", f"{r2:.3f}")
            st.metric("Explained Variance", f"{evs:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("Median Abs. Error", f"{medae:.2f}")

    with st.expander("Metric Explanations"):
        st.markdown("""
        - **R-squared / Adj. R-squared:** Proportion of variance in the target that is predictable from the features. Adjusted R² is preferred for Linear Regression as it penalizes adding useless features.
        - **Explained Variance Score:** Measures how well the model accounts for the dispersion of the dataset. A score of 1.0 is perfect.
        - **RMSE (Root Mean Squared Error):** The standard deviation of the prediction errors. Sensitive to large errors.
        - **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values.
        - **Median Absolute Error:** Robust to outliers, it measures the median of all absolute differences between predicted and actual values.
        """)

    st.markdown("#### Performance Visualizations")

    # Visual 1: Feature Correlation Heatmap
    corr_matrix = data[num_features].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("1. Correlation Matrix of Numeric Features")
    st.pyplot(fig_corr)
    st.markdown(
        "**Why this visual?** This heatmap is crucial for diagnosing **multicollinearity**. High correlation (values close to +1 or -1) between two predictor variables can destabilize linear models. For tree-based models like Random Forest, it's less of an issue but still indicates feature redundancy.")
    plt.close(fig_corr)

    # Visual 2: Actual vs. Predicted
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=2, color='red')
    ax1.set_xlabel("Actual Cost")
    ax1.set_ylabel("Predicted Cost")
    ax1.set_title("2. Actual vs. Predicted Plot")
    st.pyplot(fig1)
    st.markdown(
        "**Why this visual?** This is a direct assessment of prediction accuracy. The closer the points cluster around the red dashed line, the more accurate the model is.")
    plt.close(fig1)

    # Visual 3: Residuals Plot
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax2)
    ax2.hlines(0, y_pred.min(), y_pred.max(), colors='red', linestyles='--')
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals (Error)")
    ax2.set_title("3. Residuals vs. Predicted Values Plot")
    st.pyplot(fig2)
    st.markdown(
        "**Why this visual?** It helps diagnose bias and heteroscedasticity. Ideally, residuals should be randomly scattered around the zero line. If you see a pattern (e.g., a cone shape), it indicates the model's error is not consistent across all prediction ranges.")
    plt.close(fig2)

    # Visual 4: Error Distribution
    fig3, ax3 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title("4. Distribution of Prediction Errors (Residuals)")
    ax3.set_xlabel("Error Value")
    st.pyplot(fig3)
    st.markdown(
        "**Why this visual?** This checks the assumption of normally distributed errors for some models. A bell-shaped curve centered at zero suggests the model's errors are unbiased.")
    plt.close(fig3)

    # Visual 5 & 6: Feature Importance for Tree Models
    if model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
        cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
            cat_features)
        all_feature_names = np.concatenate([num_features, cat_feature_names])
        importances = pipeline.named_steps['model'].feature_importances_
        forest_importances = pd.Series(importances, index=all_feature_names).nlargest(15)

        fig4, ax4 = plt.subplots()
        forest_importances.plot(kind='barh', ax=ax4)
        ax4.set_title("5. Feature Importance Plot")
        ax4.invert_yaxis()
        st.pyplot(fig4)
        st.markdown(
            "**Why this visual?** This chart ranks the factors that most influence the model's predictions, helping to understand *what drives the cost* according to the model.")
        plt.close(fig4)

        # Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, cv=3, n_jobs=-1,
                                                                train_sizes=np.linspace(0.1, 1.0, 5), scoring="r2")
        train_scores_mean, test_scores_mean = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)
        fig5, ax5 = plt.subplots()
        ax5.grid()
        ax5.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax5.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax5.set_title("6. Learning Curve")
        ax5.set_xlabel("Training examples")
        ax5.set_ylabel("R-squared Score")
        ax5.legend(loc="best")
        st.pyplot(fig5)
        st.markdown(
            "**Why this visual?** It diagnoses bias vs. variance. If the training and validation curves converge at a low score, the model may be underfitting (high bias). If there's a large gap between them, it's likely overfitting (high variance).")
        plt.close(fig5)


# --- 2. Classification Models ---
def run_classification_models(df, model_name):
    # --- Performance Dashboard ---
    st.markdown("#### Model Performance Dashboard (based on 5-Fold Cross-Validation)")
    summary_df = _get_classification_performance_summary(df)
    st.dataframe(summary_df.set_index('Model'))

    with st.expander("About Cross-Validation"):
        st.markdown("""
        **Cross-Validation (CV)** provides a more robust measure of model performance.
        - **CV Accuracy Mean:** The average accuracy across all 5 folds.
        - **CV Accuracy Std:** A low standard deviation indicates stable performance.
        """)

    # Metric Correlation Matrix
    metric_corr = summary_df.drop(columns=['Model']).corr()
    fig_met_corr, ax_met_corr = plt.subplots()
    sns.heatmap(metric_corr, annot=True, cmap='viridis', fmt='.2f', ax=ax_met_corr)
    ax_met_corr.set_title("Correlation Matrix of Performance Metrics")
    st.pyplot(fig_met_corr)
    st.markdown(
        "**Why this visual?** This matrix shows how metrics like Accuracy, F1-score, and MCC relate to each other. A high correlation (as seen here) indicates that models that perform well on one metric tend to perform well on others for this dataset.")
    plt.close(fig_met_corr)

    fig_summary, ax_summary = plt.subplots()
    sns.barplot(x="CV Accuracy Mean", y="Model", data=summary_df.sort_values("CV Accuracy Mean", ascending=False),
                ax=ax_summary)
    ax_summary.set_title("Model Comparison by Cross-Validated Accuracy")
    st.pyplot(fig_summary)
    plt.close(fig_summary)
    st.markdown("---")

    # --- Detailed Evaluation ---
    st.markdown("#### Detailed Evaluation on a Single Train/Test Split")
    data = preprocess_data(df)
    labels = ['Lowest Cost', 'Below Average Cost', 'Above Average Cost', 'Highest Cost']
    data['Cost_Efficiency_Class'] = data.groupby('crop_name')['prod_cost_c2'].transform(
        lambda x: pd.qcut(x, q=4, labels=labels, duplicates='drop'))
    data.dropna(subset=['Cost_Efficiency_Class'], inplace=True)
    le = LabelEncoder()
    data['Cost_Efficiency_Class_Encoded'] = le.fit_transform(data['Cost_Efficiency_Class'])
    target = 'Cost_Efficiency_Class_Encoded'
    y = data[target]
    features = ['state_name', 'crop_name', 'year', 'derived_yield', 'opr_cost_seed', 'opr_cost_fertilizer',
                'total_human_labor_cost', 'total_machine_labor_cost']
    X = data[features]

    st.info(f"**Features Used for Classification:** `{'`, `'.join(features)}`")

    cat_features = ['state_name', 'crop_name'];
    num_features = [col for col in features if col not in cat_features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])

    models = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier": xgb.XGBClassifier(objective='multi:softmax', random_state=42, eval_metric='mlogloss'),
        "Support Vector Classifier (SVC)": SVC(probability=True, random_state=42)
    }
    model = models[model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    y_test_labels, y_pred_labels = le.inverse_transform(y_test), le.inverse_transform(y_pred_encoded)

    st.subheader(f"--- Detailed Evaluation for {model_name} ---")
    col1, col2 = st.columns(2)
    col1.metric("Overall Accuracy", f"{accuracy_score(y_test_labels, y_pred_labels):.1%}")
    col2.metric("Matthews Corr. Coef. (MCC)", f"{matthews_corrcoef(y_test, y_pred_encoded):.3f}")

    with st.expander("Metric Explanations"):
        st.markdown("""
        - **Accuracy:** The overall percentage of correct predictions.
        - **Matthews Correlation Coefficient (MCC):** A highly reliable metric that produces a balanced score between -1 and +1, where +1 is a perfect prediction. It's particularly useful when the classes are of very different sizes.
        """)

    st.text("Classification Report:");
    st.text(classification_report(y_test_labels, y_pred_labels, labels=labels))

    # Visual 1: Feature Correlation Heatmap
    corr_matrix = data[num_features].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("1. Correlation Matrix of Numeric Features")
    st.pyplot(fig_corr)
    st.markdown(
        "**Why this visual?** Checking for highly correlated features is a good practice to understand the relationships in the data that the model will learn from. It can highlight potential redundancies in the feature set.")
    plt.close(fig_corr)

    # Visual 2: Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels, normalize='true')
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title("2. Normalized Confusion Matrix")
    ax1.set_ylabel('Actual Label')
    ax1.set_xlabel('Predicted Label')
    st.pyplot(fig1)
    st.markdown(
        "**Why this visual?** The normalized confusion matrix shows the percentage of correct and incorrect predictions for each class. It's excellent for seeing which classes the model confuses most often.")
    plt.close(fig1)

    # Visual 3: ROC Curve
    fig2, ax2 = plt.subplots()
    for i, label in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('3. ROC Curve for each class')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
    st.markdown(
        "**Why this visual?** The ROC Curve shows the model's ability to distinguish between classes. A curve pushed towards the top-left corner indicates a better-performing model. The AUC score quantifies this.")
    plt.close(fig2)

    # Visual 4: Precision-Recall Curve
    fig3, ax3 = plt.subplots()
    for i, label in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
        ax3.plot(recall, precision, label=f'{label}')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('4. Precision-Recall Curve for each class')
    ax3.legend(loc="lower left")
    st.pyplot(fig3)
    st.markdown(
        "**Why this visual?** This is useful for evaluating performance on a per-class basis, showing the trade-off between identifying true positives (precision) and capturing all positives (recall).")
    plt.close(fig3)

    if model_name in ["Random Forest Classifier", "XGBoost Classifier"]:
        # Visual 5: Feature Importance
        cat_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features)
        all_names = np.concatenate([num_features, cat_names])
        importances = pd.Series(pipeline.named_steps['model'].feature_importances_, index=all_names).nlargest(15)
        fig4, ax4 = plt.subplots()
        importances.plot(kind='barh', ax=ax4)
        ax4.set_title("5. Feature Importance")
        ax4.invert_yaxis()
        st.pyplot(fig4)
        st.markdown(
            "**Why this visual?** It shows which factors are most decisive in classifying a farm's cost efficiency.")
        plt.close(fig4)


# --- 3. Clustering Models ---
def run_clustering_models(df, model_name):
    data = preprocess_data(df, for_clustering=True)
    features = ['derived_yield', 'net_return', 'labor_cost_ratio', 'machine_cost_ratio', 'fertilizer_cost_ratio',
                'mechanization_index']
    X = data[features]

    st.info(f"**Features Used for Clustering:** `{'`, `'.join(features)}`")

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(imputer.fit_transform(X))

    st.subheader(f"--- Analysis using {model_name} ---")

    # Visual 1: Feature Correlation Heatmap
    corr_matrix = data[features].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("1. Correlation Matrix of Clustering Features")
    st.pyplot(fig_corr)
    st.markdown(
        "**Why this visual?** Understanding the relationships between features used for clustering is key. Highly correlated features can have a disproportionate influence on distance-based algorithms like K-Means. This helps interpret the final cluster profiles.")
    plt.close(fig_corr)

    if model_name == "K-Means":
        k = st.slider("Select Number of Clusters (K)", 2, 8, 4)
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        data['cluster'] = model.fit_predict(X_processed)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score", f"{silhouette_score(X_processed, data['cluster']):.3f}")
        col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_processed, data['cluster']):.3f}")

        # Visuals for K-Means
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)
        fig1, ax1 = plt.subplots()
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=data['cluster'], cmap='viridis', alpha=0.7)
        ax1.set_title("2. 2D PCA of Clusters")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        st.pyplot(fig1)
        st.markdown(
            "**Why this visual?** It projects the complex data into two dimensions to visualize how distinct and separated the clusters are.")
        plt.close(fig1)

        # Silhouette Plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        silhouette_vals = silhouette_samples(X_processed, data['cluster'])
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(data['cluster'])):
            cluster_silhouette_vals = silhouette_vals[data['cluster'] == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax2.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax2.text(-0.03, (y_lower + y_upper) / 2, str(i))
            y_lower += len(cluster_silhouette_vals)
        avg_score = np.mean(silhouette_vals)
        ax2.axvline(avg_score, linestyle='--', linewidth=2, color='red')
        ax2.set_yticks([])
        ax2.set_xlim([-0.1, 1])
        ax2.set_xlabel("Silhouette coefficient values")
        ax2.set_ylabel("Cluster labels")
        ax2.set_title("3. Silhouette Plot for Clusters")
        st.pyplot(fig2)
        st.markdown(
            "**Why this visual?** The silhouette plot shows how well each point lies within its cluster. Wide, uniform plots for each cluster that extend beyond the average score (red line) are ideal.")
        plt.close(fig2)

    elif model_name == "DBSCAN":
        eps = st.slider("Epsilon (eps) - neighborhood distance", 0.1, 2.0, 0.75, 0.05)
        min_samples = st.slider("Minimum Samples", 2, 20, 10)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        data['cluster'] = model.fit_predict(X_processed)
        n_clusters = len(set(data['cluster'])) - (1 if -1 in data['cluster'] else 0)
        n_outliers = list(data['cluster']).count(-1)
        st.metric("Number of Clusters Found", n_clusters)
        st.metric("Number of Outliers", n_outliers)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['cluster'], palette="deep", alpha=0.7, ax=ax)
        ax.set_title("2. 2D PCA of DBSCAN Clusters (-1 = Outliers)")
        st.pyplot(fig)
        st.markdown(
            "**Why this visual?** It helps visualize the density-based clusters and identify points that don't belong to any group (outliers).")
        plt.close(fig)

    elif model_name == "Agglomerative Clustering":
        fig, ax = plt.subplots(figsize=(15, 7))
        linked = linkage(X_processed, method='ward')
        dendrogram(linked, orientation='top', p=5, truncate_mode='level', show_leaf_counts=True, ax=ax)
        plt.title("2. Hierarchical Clustering Dendrogram (Truncated)")
        st.pyplot(fig)
        st.markdown(
            "**Why this visual?** The dendrogram is the primary output, showing the tree-like structure of how data points are merged into clusters. It's excellent for understanding data taxonomy.")
        plt.close(fig)


# --- 4. Best Model Finder & Prediction Function ---
@st.cache_data
def get_best_regression_model(df):
    """
    Trains all regression models and returns the name of the one with the highest R-squared.
    """
    data = preprocess_data(df)
    target = 'prod_cost_c2'
    data.dropna(subset=[target], inplace=True)
    y = data[target]
    features = ['state_name', 'crop_name', 'year', 'derived_yield', 'opr_cost_seed', 'opr_cost_fertilizer',
                'total_human_labor_cost', 'total_machine_labor_cost']
    X = data[features]

    cat_features = ['state_name', 'crop_name']
    num_features = [col for col in features if col not in cat_features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_features),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "Support Vector Regressor (SVR)": SVR()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model_name = ""
    best_r2_score = -1

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model_name = name

    return best_model_name


def predict_cost(df, input_data, model_name='XGBoost Regressor'):
    state, crop, future_year = input_data['state_name'], input_data['crop_name'], input_data['year']
    if df[(df['state_name'] == state) & (df['crop_name'] == crop)].empty:
        return None, None
    data = preprocess_data(df)
    target = 'cul_cost_c2'
    data.dropna(subset=[target], inplace=True)
    y = data[target]
    features = ['state_name', 'crop_name', 'year']
    X = data[features]
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), ['year']),
                                                   ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                                                    ['state_name', 'crop_name'])])

    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "Support Vector Regressor (SVR)": SVR()
    }
    model = models[model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X, y)

    if model_name in ['Random Forest Regressor', 'XGBoost Regressor']:
        combo_df = data[(data['state_name'] == state) & (data['crop_name'] == crop)]
        if len(combo_df) >= 3:
            trend_model = LinearRegression().fit(combo_df[['year']], combo_df[target])
            annual_increase = trend_model.coef_[0]
            last_year = combo_df['year'].max()
            baseline_pred = \
                pipeline.predict(pd.DataFrame([{'state_name': state, 'crop_name': crop, 'year': last_year}]))[0]
            final_pred = baseline_pred + (annual_increase * (future_year - last_year))
            return max(final_pred, 0), pipeline

    prediction = pipeline.predict(pd.DataFrame([input_data]))[0]
    return max(prediction, 0), pipeline


# --- 5. Advanced Feature Functions ---
def run_what_if_analysis(df):
    st.info(
        "Select a state and crop to load the latest average cost data. Then, use the sliders to simulate cost changes.")
    col1, col2 = st.columns(2)
    with col1:
        states = sorted(df['state_name'].unique())
        state = st.selectbox("Select State", states, key="whatif_state")
    with col2:
        crops = sorted(df[df['state_name'] == state]['crop_name'].unique())
        if not crops:
            st.warning(f"No crop data available for {state}.")
            return
        crop = st.selectbox("Select Crop", crops, key="whatif_crop")
    filtered_data = df[(df['state_name'] == state) & (df['crop_name'] == crop)]
    if filtered_data.empty:
        st.warning(f"No data available for {crop} in {state}.")
        return
    latest_year = filtered_data['year'].max()
    latest_data = filtered_data[filtered_data['year'] == latest_year].iloc[0]
    st.markdown(f"#### Baseline Costs for **{crop}** in **{state}** (Year: {latest_year})")
    base_costs = {'Human Labor': latest_data.get('total_human_labor_cost', 0),
                  'Machine Labor': latest_data.get('total_machine_labor_cost', 0),
                  'Fertilizer': latest_data.get('opr_cost_fertilizer', 0), 'Seeds': latest_data.get('opr_cost_seed', 0),
                  'Insecticides': latest_data.get('opr_cost_insecticides', 0)}
    total_cost_base = latest_data.get('cul_cost_c2', sum(base_costs.values()))
    other_costs = max(0, total_cost_base - sum(base_costs.values()))
    st.markdown("---")
    st.sidebar.header("Cost Simulators")
    st.sidebar.markdown("Adjust the sliders below to see the impact on total costs.")
    adjustments = {}
    for cost_name, cost_value in base_costs.items():
        if cost_value > 0:
            adjustments[cost_name] = st.sidebar.slider(f"Change in {cost_name} Cost (%)", -50, 100, 0, 5,
                                                       key=f"slider_{cost_name}")
        else:
            adjustments[cost_name] = 0
    new_costs = {name: val * (1 + adj / 100.0) for (name, val), adj in zip(base_costs.items(), adjustments.values())}
    total_cost_new = sum(new_costs.values()) + other_costs
    percentage_change_total = ((total_cost_new - total_cost_base) / total_cost_base) * 100 if total_cost_base > 0 else 0
    st.subheader("Simulation Results")
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label="Original Total Cultivation Cost", value=f"₹ {total_cost_base:,.2f}")
    col_res2.metric(label="New Simulated Total Cost", value=f"₹ {total_cost_new:,.2f}",
                    delta=f"{percentage_change_total:.2f}%")
    labels = list(base_costs.keys()) + ['Other Costs']
    base_values = list(base_costs.values()) + [other_costs]
    new_values = list(new_costs.values()) + [other_costs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=base_values, name='Original Costs', marker_color='indianred'))
    fig.add_trace(go.Bar(x=labels, y=new_values, name='Simulated Costs', marker_color='lightsalmon'))
    fig.update_layout(barmode='group', title_text='Original vs. Simulated Cost Breakdown', xaxis_title="Cost Component",
                      yaxis_title="Cost (₹ per Hectare)")
    st.plotly_chart(fig, use_container_width=True)


def run_profitability_calculator(df):
    st.info(
        "First, get a cost prediction for next year. Then, enter your expected yield and market price to see the potential profit.")
    st.subheader("Step 1: Predict Your Cultivation Cost")
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("Select State", sorted(df['state_name'].unique()), key="profit_state")
    with col2:
        crop = st.selectbox("Select Crop", sorted(df['crop_name'].unique()), key="profit_crop")
    next_year = pd.to_datetime('today').year + 1
    predicted_cost, _ = predict_cost(df, {'state_name': state, 'crop_name': crop, 'year': next_year})
    if predicted_cost is None:
        st.warning(f"No historical data available for {crop} in {state}. Cannot calculate profitability.")
        return
    st.metric(label=f"Predicted Cultivation Cost for {next_year}", value=f"₹ {predicted_cost:,.2f} per Hectare")
    st.markdown("---")
    st.subheader("Step 2: Enter Your Expectations")
    col3, col4 = st.columns(2)
    avg_yield = df[(df['state_name'] == state) & (df['crop_name'] == crop)]['derived_yield'].mean()
    avg_yield = float(round(avg_yield, 2)) if pd.notna(avg_yield) and avg_yield > 0 else 10.0
    with col3:
        expected_yield = st.number_input("Expected Yield (Quintals per Hectare)", min_value=0.0, value=avg_yield,
                                         step=1.0)
    with col4:
        expected_price = st.number_input("Expected Market Price (₹ per Quintal)", min_value=0, value=2000, step=100)
    if expected_yield > 0:
        total_revenue, net_profit = expected_yield * expected_price, (expected_yield * expected_price) - predicted_cost
        break_even_price = predicted_cost / expected_yield
        st.markdown("---")
        st.subheader("Financial Outlook")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Total Revenue", f"₹ {total_revenue:,.2f}")
        res_col2.metric("Predicted Net Profit / Loss", f"₹ {net_profit:,.2f}")
        res_col3.metric("Break-Even Price", f"₹ {break_even_price:,.2f} / Quintal")
        fig = go.Figure(
            go.Waterfall(name="Profit Breakdown", orientation="v", measure=["relative", "relative", "total"],
                         x=["Total Revenue", "Cultivation Cost", "Net Profit"], textposition="outside",
                         text=[f"₹{total_revenue:,.0f}", f"₹{-predicted_cost:,.0f}", f"₹{net_profit:,.0f}"],
                         y=[total_revenue, -predicted_cost, net_profit],
                         connector={"line": {"color": "rgb(63, 63, 63)"}}, ))
        fig.update_layout(title="Profitability Breakdown (per Hectare)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

