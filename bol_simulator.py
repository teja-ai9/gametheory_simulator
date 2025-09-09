import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Barilla Promo Strategy – Game Theory Simulator", layout="wide")

# ========== SIDEBAR ==========
st.sidebar.markdown("### Barilla – Commercial Data Science Demo\n**Price & Promo Strategy (Game Theory)**\nby **Teja Bonthalakoti**")
st.sidebar.caption("Assumptions are illustrative and configurable for a live demo.")

st.sidebar.header("Category & Economics (per pack)")
category = st.sidebar.selectbox("Category / Use-case", ["Pasta (Core)", "Pasta (Premium)", "Sauces"])
base_demand = st.sidebar.slider("Base Demand (units)", 200, 20000, 6000, 100)
wholesale_price = st.sidebar.slider("Barilla Wholesale Price to Retailer (€)", 0.50, 3.50, 1.30, 0.05)
cogs = st.sidebar.slider("Barilla COGS (€)", 0.20, 1.20, 0.55, 0.01)

st.sidebar.header("Retail & Shopper")
retail_margin_pct = st.sidebar.slider("Retailer Margin on RSP (%)", 10, 40, 25, 1) / 100
feature_display_cost = st.sidebar.slider("Feature/Display Cost per unit (€)", 0.00, 0.50, 0.05, 0.01)
pass_through = st.sidebar.slider("Promo Pass-through to Shelf (%)", 50, 100, 80, 5) / 100
stockout_penalty = st.sidebar.slider("Penalty if Barilla price > Competitor by >5% (multiplier)", 0.30, 1.00, 0.70, 0.05)

st.sidebar.header("Elasticities")
price_elasticity_barilla = st.sidebar.slider("Barilla Own Price Elasticity", -5.0, -0.5, -2.2, 0.1)
cross_elast_comp_on_barilla = st.sidebar.slider("Cross Elasticity (Competitor→Barilla)", 0.0, 3.0, 0.60, 0.05)

price_elasticity_comp = st.sidebar.slider("Competitor Own Price Elasticity", -5.0, -0.5, -1.8, 0.1)
cross_elast_barilla_on_comp = st.sidebar.slider("Cross Elasticity (Barilla→Competitor)", 0.0, 3.0, 0.50, 0.05)

st.sidebar.header("Competitor (e.g., Private Label)")
comp_wholesale_price = st.sidebar.slider("Competitor Wholesale Price (€)", 0.40, 3.00, 1.00, 0.05)
comp_cogs = st.sidebar.slider("Competitor COGS (€)", 0.15, 1.00, 0.45, 0.01)

st.sidebar.header("Promo Grids")
barilla_trade_disc_grid = np.round(np.arange(0.00, 0.90, 0.05), 2)  # € discount funded by Barilla
comp_trade_disc_grid = np.round(np.arange(0.00, 0.90, 0.05), 2)      # € discount funded by competitor

# Helper to compute RSP given wholesale and retailer margin
def rsp_from_wholesale(wh, margin_pct):
    # RSP = wholesale / (1 - margin%)
    return wh / (1 - margin_pct)

# Base (no-promo) shelf prices
barilla_base_rsp = rsp_from_wholesale(wholesale_price, retail_margin_pct)
comp_base_rsp = rsp_from_wholesale(comp_wholesale_price, retail_margin_pct)

# ========== SIMULATION ==========
rows = []
for bar_disc in barilla_trade_disc_grid:
    for comp_disc in comp_trade_disc_grid:
        # Pass-through to shelf
        barilla_shelf_cut = bar_disc * pass_through
        comp_shelf_cut = comp_disc * pass_through

        barilla_rsp = max(0.01, barilla_base_rsp - barilla_shelf_cut)
        comp_rsp = max(0.01, comp_base_rsp - comp_shelf_cut)

        # Relative price penalty if Barilla > Competitor by >5%
        ratio = barilla_rsp / comp_rsp if comp_rsp > 0 else 1.0
        penalty = stockout_penalty if ratio > 1.05 else 1.0

        # Demand models (own & cross effects)
        # Normalize to base rsp anchors
        barilla_demand = base_demand \
            * (barilla_rsp / barilla_base_rsp) ** (price_elasticity_barilla) \
            * (comp_rsp / comp_base_rsp) ** (cross_elast_comp_on_barilla) \
            * penalty

        comp_demand = base_demand \
            * (comp_rsp / comp_base_rsp) ** (price_elasticity_comp) \
            * (barilla_rsp / barilla_base_rsp) ** (cross_elast_barilla_on_comp)

        # Manufacturer (Barilla) economics
        # Trade spend is the *full* bar_disc per unit (what Barilla funds); retailer passes pass_through to shelf
        barilla_trade_spend = bar_disc * barilla_demand
        barilla_nsv = (wholesale_price - bar_disc) * barilla_demand  # net sales after trade discount
        barilla_gross_margin = (wholesale_price - bar_disc - cogs) * barilla_demand
        barilla_feature_cost = feature_display_cost * barilla_demand
        barilla_profit = barilla_gross_margin - barilla_feature_cost

        # Retailer margin (for Barilla units)
        retailer_margin_per_unit_bar = barilla_rsp - (wholesale_price - bar_disc)
        retailer_margin_bar = retailer_margin_per_unit_bar * barilla_demand

        # Competitor manufacturer & retailer
        comp_trade_spend = comp_disc * comp_demand
        comp_nsv = (comp_wholesale_price - comp_disc) * comp_demand
        comp_gross_margin = (comp_wholesale_price - comp_disc - comp_cogs) * comp_demand
        comp_profit = comp_gross_margin  # assume no extra feature cost for simplicity

        retailer_margin_per_unit_comp = comp_rsp - (comp_wholesale_price - comp_disc)
        retailer_margin_comp = retailer_margin_per_unit_comp * comp_demand

        rows.append({
            "Barilla Trade Disc (€)": bar_disc,
            "Competitor Trade Disc (€)": comp_disc,
            "Barilla RSP (€)": barilla_rsp,
            "Competitor RSP (€)": comp_rsp,
            "Barilla Demand (u)": barilla_demand,
            "Competitor Demand (u)": comp_demand,
            "Barilla NSV (€)": barilla_nsv,
            "Barilla Profit (€)": barilla_profit,
            "Retailer Margin – Barilla (€)": retailer_margin_bar,
            "Competitor Profit (€)": comp_profit,
            "Retailer Margin – Competitor (€)": retailer_margin_comp,
            "Retailer Total Margin (€)": retailer_margin_bar + retailer_margin_comp
        })

df = pd.DataFrame(rows)

# ========== PAYOFF MATRICES ==========
# Manufacturer profit viewpoints (Barilla vs Competitor)
barilla_matrix = df.pivot(index="Competitor Trade Disc (€)", columns="Barilla Trade Disc (€)", values="Barilla Profit (€)")
comp_matrix = df.pivot(index="Competitor Trade Disc (€)", columns="Barilla Trade Disc (€)", values="Competitor Profit (€)")

# Best responses
barilla_best_response = barilla_matrix.idxmax(axis=1)          # given competitor choice, Barilla's best trade discount
comp_best_response = comp_matrix.idxmax(axis=0)                # given Barilla choice, Competitor's best trade discount

# Nash equilibria (pure)
nash_points = []
for comp_disc in comp_trade_disc_grid:
    bar_disc = barilla_best_response[comp_disc]
    comp_best = comp_best_response[bar_disc]
    if np.isclose(comp_disc, comp_best):
        nash_points.append((comp_disc, bar_disc))

# ========== UI ==========
st.title("Barilla Price & Promo Strategy – Game Theory Simulator")

colA, colB, colC = st.columns(3)
with colA:
    best_row_bar = df.loc[df["Barilla Profit (€)"].idxmax()]
    st.metric("Max Barilla Profit (€)", f"{best_row_bar['Barilla Profit (€)']:.0f}")
with colB:
    best_barilla_disc_avg = df.groupby("Barilla Trade Disc (€)")["Barilla Profit (€)"].mean().idxmax()
    st.metric("Recommended Barilla Trade Discount (avg across competitor moves)", f"€{best_barilla_disc_avg:.2f}")
with colC:
    best_row_retailer = df.loc[df["Retailer Total Margin (€)"].idxmax()]
    st.metric("Max Retailer Total Margin (€)", f"{best_row_retailer['Retailer Total Margin (€)']:.0f}")

st.subheader("Barilla Profit Payoff Matrix (€)")
fig1, ax1 = plt.subplots(figsize=(12, 7))
sns.heatmap(barilla_matrix, annot=False, fmt=".0f", cmap="YlGnBu", ax=ax1)
for (comp_disc, bar_disc) in nash_points:
    x_idx = list(barilla_matrix.columns).index(bar_disc)
    y_idx = list(barilla_matrix.index).index(comp_disc)
    ax1.plot(x_idx + 0.5, y_idx + 0.5, 'ro', markersize=10)
ax1.set_xlabel("Barilla Trade Discount (€)")
ax1.set_ylabel("Competitor Trade Discount (€)")
ax1.set_title("Barilla Manufacturer Profit")
st.pyplot(fig1)

st.subheader("Retailer Total Margin (€) – Outcome Surface")
retailer_matrix = df.pivot(index="Competitor Trade Disc (€)", columns="Barilla Trade Disc (€)", values="Retailer Total Margin (€)")
fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.heatmap(retailer_matrix, annot=False, fmt=".0f", cmap="Greys", ax=ax2)
ax2.set_xlabel("Barilla Trade Discount (€)")
ax2.set_ylabel("Competitor Trade Discount (€)")
ax2.set_title("Retailer Margin Landscape")
st.pyplot(fig2)

st.subheader("Nash Equilibrium (pure strategies)")
if nash_points:
    for comp_disc, bar_disc in nash_points:
        sub = df[(df["Competitor Trade Disc (€)"] == comp_disc) & (df["Barilla Trade Disc (€)"] == bar_disc)].iloc[0]
        st.info(
            f"Competitor Trade Disc: €{comp_disc:.2f} | Barilla Trade Disc: €{bar_disc:.2f} "
            f"→ Barilla Profit: €{sub['Barilla Profit (€)']:.0f}, Retailer Total Margin: €{sub['Retailer Total Margin (€)']:.0f}"
        )
else:
    st.warning("No pure-strategy Nash equilibrium found under current assumptions.")

with st.expander("See sample rows"):
    st.dataframe(df.head(20))

st.subheader("Download Simulation Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="barilla_promo_game_simulation.csv",
    mime="text/csv"
)

st.caption(
    "Notes: Demand response uses constant elasticities (own & cross). Trade discount is manufacturer-funded; pass-through sets how much reaches shelf. "
    "Retailer margin uses a % of RSP framing via margin-on-price transformation. This is illustrative and can be calibrated with real price tests / MMM / uplift studies."
)
