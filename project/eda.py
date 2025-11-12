import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_data
def perform_eda(df):
    """
    Performs and displays Exploratory Data Analysis visuals in Streamlit.

    Args:
        df (pd.DataFrame): The pre-loaded and pre-processed dataframe.
    """

    # Chart 1: Distribution of Data by Crop Type
    st.subheader("1. Distribution of Data by Crop Type")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    df['crop_type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis', ax=ax1)
    ax1.set_title('Distribution of Data by Crop Type')
    ax1.set_ylabel('')
    st.pyplot(fig1)
    st.markdown(
        "This chart shows the proportion of data entries for each major crop category. Cereals and Pulses make up the largest portions.")

    # Chart 2: Number of Data Entries per State
    st.subheader("2. Number of Data Entries per State")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.countplot(y='state_name', data=df, order=df['state_name'].value_counts().index, palette='viridis', ax=ax2,
                  hue='state_name', legend=False)
    ax2.set_title('Number of Data Entries per State')
    ax2.set_xlabel('Number of Entries')
    ax2.set_ylabel('State')
    st.pyplot(fig2)
    st.markdown(
        "This bar chart displays the geographic distribution of the dataset, showing how many agricultural data entries come from each Indian state.")

    # Chart 3: Average Total Cultivation Cost (C2) by State
    st.subheader("3. Average Total Cultivation Cost (C2) by State")
    avg_cost_state = df.groupby('state_name')['cul_cost_c2'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=avg_cost_state.index, x=avg_cost_state.values, palette='viridis_r', ax=ax3, hue=avg_cost_state.index,
                legend=False)
    ax3.set_title('Average Total Cultivation Cost (C2) by State')
    ax3.set_xlabel('Average Cost (INR per Hectare)')
    ax3.set_ylabel('State')
    st.pyplot(fig3)
    st.markdown(
        "This chart ranks states by their average total cost of cultivation, revealing significant regional differences in farming expenses.")

    # Chart 4: Top 10 Costliest Crops to Cultivate (Cost C2)
    st.subheader("4. Top 10 Costliest Crops to Cultivate")
    avg_cost_crop = df.groupby('crop_name')['cul_cost_c2'].mean().nlargest(10)
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=avg_cost_crop.index, x=avg_cost_crop.values, palette='rocket_r', ax=ax4, hue=avg_cost_crop.index,
                legend=False)
    ax4.set_title('Top 10 Costliest Crops to Cultivate (Cost C2)')
    ax4.set_xlabel('Average Cost (INR per Hectare)')
    ax4.set_ylabel('Crop')
    st.pyplot(fig4)
    st.markdown(
        "This identifies the ten crops with the highest average cultivation cost. High-value crops like Sugarcane and Onion are capital-intensive to grow.")

    # Chart 5: Cultivation Cost vs. Yield
    st.subheader("5. Cultivation Cost vs. Yield")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='cul_cost_c2', y='derived_yield', hue='crop_type', alpha=0.6, ax=ax5)
    ax5.set_title('Cultivation Cost vs. Yield')
    ax5.set_xlabel('Total Cultivation Cost (INR per Hectare)')
    ax5.set_ylabel('Yield (Quintal per Hectare)')
    st.pyplot(fig5)
    st.markdown(
        "This scatter plot explores the relationship between total cultivation cost and yield. Generally, higher investment leads to higher yield, but there is considerable variation.")

    # Ensure net_return is calculated for subsequent charts
    if 'net_return' not in df.columns:
        df['net_return'] = (df['main_product_value'] + df['by_product_value']) - df['cul_cost_c2']

    # Chart 6: Distribution of Net Return
    st.subheader("6. Distribution of Net Return (Profit/Loss)")
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    sns.histplot(df['net_return'], bins=30, kde=True, color='green', ax=ax6)
    ax6.axvline(0, color='red', linestyle='--', label='Break-even Point')
    ax6.set_title('Distribution of Net Return')
    ax6.set_xlabel('Net Return (INR per Hectare)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    st.pyplot(fig6)
    st.markdown(
        "This histogram shows the distribution of profit or loss. A significant portion falls to the left of the red break-even line, indicating that many farming operations in the dataset resulted in a financial loss.")

    # Chart 7: Top 10 Most Profitable Crops
    st.subheader("7. Top 10 Most Profitable Crops")
    avg_profit_crop = df.groupby('crop_name')['net_return'].mean().nlargest(10)
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=avg_profit_crop.index, x=avg_profit_crop.values, palette='summer', ax=ax7, hue=avg_profit_crop.index,
                legend=False)
    ax7.set_title('Top 10 Most Profitable Crops')
    ax7.set_xlabel('Average Net Return (INR per Hectare)')
    ax7.set_ylabel('Crop')
    st.pyplot(fig7)
    st.markdown(
        "This chart highlights the ten crops with the highest average net return (profit). Commercial crops like Sugarcane and Onion are often the most profitable.")

    # Chart 8: Average Net Return by State
    st.subheader("8. Average Net Return by State")
    avg_profit_state = df.groupby('state_name')['net_return'].mean().sort_values(ascending=False)
    fig8, ax8 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=avg_profit_state.index, x=avg_profit_state.values, palette='coolwarm', ax=ax8,
                hue=avg_profit_state.index, legend=False)
    ax8.set_title('Average Net Return by State')
    ax8.set_xlabel('Average Net Return (INR per Hectare)')
    ax8.set_ylabel('State')
    ax8.axvline(0, color='black', linestyle='--')
    st.pyplot(fig8)
    st.markdown(
        "This chart ranks states by their average net return from agriculture. It reveals which states have, on average, more profitable farming sectors, while some states show an average net loss.")

    # Chart 9: Trend of Costs and Returns Over Time
    st.subheader("9. Trend of Costs and Returns Over Time")
    time_trend = df.groupby('year')[['cul_cost_c2', 'net_return']].mean()
    fig9, ax9 = plt.subplots(figsize=(14, 7))
    time_trend.plot(ax=ax9, marker='o', linestyle='-')
    ax9.set_title('Trend of Average Costs and Net Returns Over Time')
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Amount (INR per Hectare)')
    ax9.grid(True)
    st.pyplot(fig9)
    st.markdown(
        "This line chart tracks the average cultivation cost and net return over the years. It clearly illustrates that while costs have risen steadily, net returns have been far more volatile and have not kept pace.")

    # Chart 10: Correlation Heatmap
    st.subheader("10. Correlation Between Key Variables")
    corr_features = ['cul_cost_c2', 'derived_yield', 'main_product_value', 'net_return', 'total_human_labor_cost',
                     'total_machine_labor_cost', 'opr_cost_fertilizer']
    # Ensure engineered columns exist before correlation
    if 'total_human_labor_cost' not in df.columns:
        df['total_human_labor_cost'] = df[
            ['opr_cost_hmn_lab_family', 'opr_cost_hmn_lab_attached', 'opr_cost_hmn_lab_casual']].sum(axis=1)
    if 'total_machine_labor_cost' not in df.columns:
        df['total_machine_labor_cost'] = df[['opr_cost_mch_lab_hired', 'opr_cost_mch_lab_owned']].sum(axis=1)

    corr_matrix = df[corr_features].corr()
    fig10, ax10 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax10)
    ax10.set_title('Correlation Matrix of Key Variables')
    st.pyplot(fig10)
    st.markdown(
        "This heatmap shows how different variables relate to each other. Red indicates a positive correlation (as one goes up, the other tends to go up), and blue indicates a negative correlation.")

