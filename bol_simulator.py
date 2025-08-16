import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar: Team Info
st.sidebar.markdown("""
### Strategic Management meets Data Science 
**by Teja Bonthalakoti**
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
mrp = st.sidebar.slider("MRP (€)", 10.0, 20.0, 15.0, 0.5)
kelloggs_promo = st.sidebar.slider("Company's Promo (€)", 0.0, 5.0, 2.0, 0.5)
commission_rate = st.sidebar.slider("Commission Rate (%)", 5, 20, 10, 1) / 100
base_demand = st.sidebar.slider("Base Demand", 0, 2000, 1000, 100)
elasticity = st.sidebar.slider("Price Elasticity", -10.0, -0.5, -3.0, 0.1)
cross_elasticity = st.sidebar.slider("Cross Elasticity", 0.0, 10.0, 1.5, 0.1)

# Promo ranges 
bol_promo_values = np.round(np.arange(0, 5, 0.5), 2)
competitor_price_cuts = np.round(np.arange(0, 5, 0.5), 2)

# Realistic assumptions clearly defined
competitor_margin = 0.10  # realistic competitor profit margin assumption

simulation_data = []
for bol_promo in bol_promo_values:
    for competitor_cut in competitor_price_cuts:
        bol_final_price = mrp - kelloggs_promo - bol_promo
        competitor_final_price = mrp - competitor_cut

        # Demand clearly adjusted based on price difference
        price_ratio = bol_final_price / competitor_final_price
        penalty = 0.7 if price_ratio > 1.05 else 1.0

        # Bol demand
        bol_demand = base_demand * ((bol_final_price / (mrp - kelloggs_promo)) ** elasticity)
        bol_demand *= (competitor_final_price / mrp) ** cross_elasticity
        bol_demand *= penalty

        bol_revenue = bol_final_price * commission_rate * bol_demand
        bol_cost = bol_promo * bol_demand * 0.5  # realistic 50% promo cost
        bol_profit = bol_revenue - bol_cost

        # Competitor demand
        comp_demand = base_demand * ((competitor_final_price / mrp) ** elasticity)
        comp_demand *= (bol_final_price / mrp) ** cross_elasticity

        competitor_profit = competitor_final_price * competitor_margin * comp_demand

        simulation_data.append({
            "Bol Promo (€)": bol_promo,
            "Amazon Promo (€)": competitor_cut,
            "Bol Revenue (€)": bol_revenue,
            "Bol Profit (€)": bol_profit,
            "Amazon Profit (€)": competitor_profit
        })

df_full_simulation = pd.DataFrame(simulation_data)

# Payoff matrices clearly created
bol_matrix = df_full_simulation.pivot(index='Amazon Promo (€)', columns='Bol Promo (€)', values='Bol Profit (€)')
competitor_matrix = df_full_simulation.pivot(index='Amazon Promo (€)', columns='Bol Promo (€)', values='Amazon Profit (€)')

# Nash Equilibrium logic (profit vs profit realistically)
bol_best_response = bol_matrix.idxmax(axis=1)
competitor_best_response = competitor_matrix.idxmax(axis=0)

nash_points = []
for comp_cut in competitor_price_cuts:
    bol_promo = bol_best_response[comp_cut]
    optimal_comp_cut = competitor_best_response[bol_promo]
    if comp_cut == optimal_comp_cut:
        nash_points.append((comp_cut, bol_promo))

# Streamlit Layout
st.title("Bol Promo Optimization - Game Theory Simulator")

# Payoff Matrix Heatmap
st.subheader("Bol Profit Payoff Matrix (€)")
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(bol_matrix, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)

# Highlight Nash Equilibrium points clearly
for (y, x) in nash_points:
    x_idx = list(bol_matrix.columns).index(x)
    y_idx = list(bol_matrix.index).index(y)
    ax.plot(x_idx + 0.5, y_idx + 0.5, 'ro', markersize=12)

ax.set_xlabel("Bol Promo (€)")
ax.set_ylabel("Amazon Promo (€)")
st.pyplot(fig)

# Summary KPIs (simple and clear)
st.subheader("Summary KPIs")
best_row = df_full_simulation.loc[df_full_simulation['Bol Profit (€)'].idxmax()]
#st.metric("Best Bol Promo (€)", f"{best_row['Bol Promo (€)']:.2f}")
st.metric("Max Profit (€)", f"{best_row['Bol Profit (€)']:.0f}")

# Recommended Bol Promo (average optimal scenario)
st.subheader("Recommended Bol Promo")
best_bol_promo = df_full_simulation.groupby("Bol Promo (€)")["Bol Profit (€)"].mean().idxmax()
st.success(f"Optimal Promo (avg. across scenarios): €{best_bol_promo:.2f}")

# Nash Equilibria displayed clearly
st.subheader("Nash Equilibrium Points (Profit-Based)")
if nash_points:
    for comp_cut, bol_promo in nash_points:
        st.info(f"Amazon Promo: €{comp_cut:.2f}, Optimal Bol Promo: €{bol_promo:.2f}")
else:
    st.warning("No Nash Equilibrium found with current inputs.")

# Download CSV (full results)
st.subheader("Download Simulation Data")
st.download_button(
    label="Download CSV",
    data=df_full_simulation.to_csv(index=False).encode('utf-8'),
    file_name='bol_promo_simulation.csv',
    mime='text/csv'
)
