import streamlit as st
import pandas as pd
import numpy as np
from engine_backend import run_risk_engine

# Configure the Streamlit page
st.set_page_config(page_title="XVA Risk Engine", layout="wide")

st.title("Multi-Asset Stochastic Pricing & XVA Risk Engine")
st.markdown("### Counterparty Credit Risk & Portfolio Optimization")

# ----------------- SESSION STATE & LOG RETENTION -----------------
if "custom_weights_str" not in st.session_state:
    st.session_state.custom_weights_str = ""

if "execution_log" not in st.session_state:
    st.session_state.execution_log = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None

def randomize_weights(num_assets):
    raw_weights = np.random.rand(num_assets)
    normalized_weights = raw_weights / np.sum(raw_weights)
    st.session_state.custom_weights_str = ", ".join([f"{w:.4f}" for w in normalized_weights])

# ----------------- SIDEBAR CONFIG -----------------
st.sidebar.header("Engine Configuration")

default_tickers = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, INFY.NS, SBIN.NS, BHARTIARTL.NS, ITC.NS, LT.NS, AXISBANK.NS, KOTAKBANK.NS, HINDUNILVR.NS, SUNPHARMA.NS, TMCV.NS, WIPRO.NS, ^NSEI, ^NSEBANK, LIQUIDBEES.NS, GOLDBEES.NS, CPSEETF.NS"
tickers_input = st.sidebar.text_input("Asset Tickers (comma separated)", default_tickers)
tickers_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
num_assets = len(tickers_list)

capital = st.sidebar.number_input(
    "Initial Portfolio Capital (INR)", 
    min_value=100000, 
    value=10000000, 
    step=100000
)

st.sidebar.markdown("### Portfolio Allocation")
if st.sidebar.button("🎲 Randomize Weights"):
    randomize_weights(num_assets)

weights_input = st.sidebar.text_area(
    f"Custom Weights (Requires {num_assets} values)",
    value=st.session_state.custom_weights_str,
    help="Enter comma-separated decimals. Leave empty to use default equal-weighting."
)

scenario_selected = st.sidebar.selectbox(
    "Macro Scenario & Derivative Structure",
    [
        "Baseline (Call Option, Normal Drift)", 
        "Bullish (Call Option, Low Volatility)", 
        "Market Crash (Put Option, Jump Shocks)"
    ]
)

scenario_map = {
    "Baseline (Call Option, Normal Drift)": "baseline",
    "Bullish (Call Option, Low Volatility)": "bullish",
    "Market Crash (Put Option, Jump Shocks)": "market_crash"
}

# ----------------- EXECUTION PIPELINE -----------------
if st.sidebar.button("Run Risk Engine"):
    scenario_name = scenario_map[scenario_selected]
    
    # Process Custom Weights Input
    custom_weights = None
    if weights_input.strip():
        try:
            custom_weights = [float(w.strip()) for w in weights_input.split(",")]
            if len(custom_weights) != num_assets:
                st.error(f"Error: You provided {len(custom_weights)} weights, but there are {num_assets} tickers.")
                st.stop()
        except ValueError:
            st.error("Error: Invalid weights format. Please provide comma-separated numeric values.")
            st.stop()
            
    st.session_state.execution_log = [] # Clear previous log for new run

    # Live Status Container for Real-Time Execution Feedback
    with st.status(f"Executing {scenario_name.upper()} Pipeline...", expanded=True) as status:
        status_container = st.empty()
        
        def update_ui_status(message):
            st.session_state.execution_log.append(message)
            status_container.markdown("\n\n".join(st.session_state.execution_log))

        # Execute Backend with Callback
        metrics, allocations, figs = run_risk_engine(
            tickers=tickers_list, 
            initial_capital=capital, 
            custom_weights=custom_weights,
            scenario_name=scenario_name,
            progress_callback=update_ui_status
        )
        
        st.session_state.last_results = {
            "metrics": metrics,
            "allocations": allocations,
            "figs": figs
        }
        
        status.update(label="Simulation & Optimization Complete!", state="complete", expanded=False)

# ----------------- PERSISTENT UI DASHBOARD RENDERING -----------------
if st.session_state.last_results is not None:
    res = st.session_state.last_results
    metrics = res["metrics"]
    allocations = res["allocations"]
    figs = res["figs"]

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Portfolio Metrics")
        st.metric("95% PFE (Peak Risk)", f"₹ {metrics['baseline']['PFE']:,.2f}")
        st.metric("Expected Exposure (EE)", f"₹ {metrics['baseline']['EE']:,.2f}")
        st.metric("CVA (Credit Value Adj.)", f"₹ {metrics['baseline']['CVA']:,.2f}")
        st.metric("FVA (Funding Value Adj.)", f"₹ {metrics['baseline']['FVA']:,.2f}")
        
    with col2:
        st.subheader("Optimized Portfolio Metrics")
        st.metric("95% PFE (Peak Risk)", f"₹ {metrics['optimized']['PFE']:,.2f}")
        st.metric("Expected Exposure (EE)", f"₹ {metrics['optimized']['EE']:,.2f}")
        st.metric("CVA (Credit Value Adj.)", f"₹ {metrics['optimized']['CVA']:,.2f}")
        st.metric("FVA (Funding Value Adj.)", f"₹ {metrics['optimized']['FVA']:,.2f}")
        
    # Display Generated Visualizations
    st.markdown("---")
    st.pyplot(figs[0])  
    st.markdown("---")
    st.pyplot(figs[1])  
    st.markdown("---")
    st.pyplot(figs[2])  
    
    # Display Capital Allocation Table with Position Sizes
    st.markdown("---")
    st.subheader("Portfolio Capital Allocation")
    
    alloc_df = pd.DataFrame(allocations, columns=["Ticker", "Baseline Weight", "Optimized Weight"])
    
    # Calculate exact INR position sizes
    alloc_df["Baseline Position (INR)"] = alloc_df["Baseline Weight"] * capital
    alloc_df["Optimized Position (INR)"] = alloc_df["Optimized Weight"] * capital
    
    # Format layout for readability
    alloc_df["Baseline Weight"] = alloc_df["Baseline Weight"].apply(lambda x: f"{x*100:.2f}%")
    alloc_df["Optimized Weight"] = alloc_df["Optimized Weight"].apply(lambda x: f"{x*100:.2f}%")
    alloc_df["Baseline Position (INR)"] = alloc_df["Baseline Position (INR)"].apply(lambda x: f"₹ {x:,.2f}")
    alloc_df["Optimized Position (INR)"] = alloc_df["Optimized Position (INR)"].apply(lambda x: f"₹ {x:,.2f}")
    
    alloc_df = alloc_df[["Ticker", "Baseline Weight", "Baseline Position (INR)", "Optimized Weight", "Optimized Position (INR)"]]
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    # Persistent Execution Log Box
    if st.session_state.execution_log:
        with st.expander("View Full Execution Log", expanded=False):
            for log_entry in st.session_state.execution_log:
                st.text(log_entry)