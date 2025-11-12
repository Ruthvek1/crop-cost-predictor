import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def run_what_if_analysis(df):
    """
    Creates an interactive UI for users to simulate the effect of changing input costs on the total cultivation cost.
    """
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

    # Filter data for the most recent year available for the selected combination
    filtered_data = df[(df['state_name'] == state) & (df['crop_name'] == crop)]
    if filtered_data.empty:
        st.warning(f"No data available for {crop} in {state}.")
        return

    latest_year = filtered_data['year'].max()
    latest_data = filtered_data[filtered_data['year'] == latest_year].iloc[0]

    st.markdown(f"#### Baseline Costs for **{crop}** in **{state}** (Year: {latest_year})")

    # --- Define Key Cost Components ---
    base_costs = {
        'Human Labor': latest_data.get('total_human_labor_cost', 0),
        'Machine Labor': latest_data.get('total_machine_labor_cost', 0),
        'Fertilizer': latest_data.get('opr_cost_fertilizer', 0),
        'Seeds': latest_data.get('opr_cost_seed', 0),
        'Insecticides': latest_data.get('opr_cost_insecticides', 0)
    }

    # Calculate "Other Costs" which is the total cost minus our key components
    total_cost_base = latest_data.get('cul_cost_c2', sum(base_costs.values()))
    other_costs = total_cost_base - sum(base_costs.values())
    if other_costs < 0: other_costs = 0  # Ensure it's not negative

    st.markdown("---")
    st.sidebar.header("Cost Simulators")
    st.sidebar.markdown("Adjust the sliders below to see the impact on total costs.")

    # --- Interactive Sliders ---
    adjustments = {}
    for cost_name, cost_value in base_costs.items():
        if cost_value > 0:  # Only show sliders for costs that are not zero
            adjustments[cost_name] = st.sidebar.slider(
                f"Change in {cost_name} Cost (%)",
                min_value=-50, max_value=100, value=0, step=5,
                key=f"slider_{cost_name}"
            )
        else:
            adjustments[cost_name] = 0

    # --- Calculate New Costs ---
    new_costs = {}
    for cost_name, base_value in base_costs.items():
        percentage_change = adjustments[cost_name] / 100.0
        new_costs[cost_name] = base_value * (1 + percentage_change)

    total_cost_new = sum(new_costs.values()) + other_costs
    percentage_change_total = ((total_cost_new - total_cost_base) / total_cost_base) * 100 if total_cost_base > 0 else 0

    # --- Display Results ---
    st.subheader("Simulation Results")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(
            label="Original Total Cultivation Cost",
            value=f"₹ {total_cost_base:,.2f}"
        )
    with col_res2:
        st.metric(
            label="New Simulated Total Cost",
            value=f"₹ {total_cost_new:,.2f}",
            delta=f"{percentage_change_total:.2f}%"
        )

    # --- Visualization ---
    st.markdown("#### Cost Component Breakdown")

    # Data for Plotly chart
    labels = list(base_costs.keys()) + ['Other Costs']
    base_values = list(base_costs.values()) + [other_costs]
    new_values = list(new_costs.values()) + [other_costs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=base_values,
        name='Original Costs',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=labels,
        y=new_values,
        name='Simulated Costs',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        barmode='group',
        title_text='Original vs. Simulated Cost Breakdown',
        xaxis_title="Cost Component",
        yaxis_title="Cost (₹ per Hectare)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "**Why this visual?** This chart provides an immediate comparison of how your adjustments to individual cost components affect the overall cost structure.")
