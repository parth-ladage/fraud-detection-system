import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Fraud Operations Dashboard", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model_artifacts():
    with open('model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_and_score_data():
    # Load dataset
    df = pd.read_parquet('sample_data.parquet')
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    threshold = artifacts['threshold']
    
    # Prepare features for prediction
    X = df[features]
    X_scaled = scaler.transform(X).astype('float32')
    
    # Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    df['FraudProbability'] = probs
    
    # Assign risk tiers
    def assign_tier(prob):
        if prob >= 0.75: return 'Critical Risk'
        elif prob >= 0.40: return 'Suspicious'
        else: return 'Clear'
        
    df['RiskTier'] = df['FraudProbability'].apply(assign_tier)
    df['PredictedFraud'] = (df['FraudProbability'] >= threshold).astype(int)
    
    # Get original transaction amounts if available, otherwise just use what we have
    if 'TransactionAmt' not in df.columns and 'TransactionAmt' in features:
        df['TransactionAmt'] = X['TransactionAmt']
        
    return df, X_scaled

# Load resources
artifacts = load_model_artifacts()
df, X_scaled = load_and_score_data()

# Global Header
st.markdown("<h1 style='text-align: center;'>Real-Time Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>with Explainable AI & Live Dashboard</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("🛡️ Fraud Operations")
page = st.sidebar.radio("Navigation", ["Overview", "Transaction Explorer", "SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
selected_tiers = st.sidebar.multiselect(
    "Filter by Risk Tier", 
    options=['Critical Risk', 'Suspicious', 'Clear'],
    default=['Critical Risk', 'Suspicious', 'Clear']
)

# Apply global filter
filtered_df = df[df['RiskTier'].isin(selected_tiers)]

if page == "Overview":
    st.title("📊 System Overview")
    
    # Calculate KPIs
    total_tx = len(filtered_df)
    total_fraud = filtered_df['isFraud'].sum()
    
    # Detection rate = True Positives / Total Actual Frauds
    actual_frauds = filtered_df[filtered_df['isFraud'] == 1]
    if len(actual_frauds) > 0:
        detected_frauds = actual_frauds['PredictedFraud'].sum()
        detection_rate = detected_frauds / len(actual_frauds)
    else:
        detection_rate = 0.0
        
    avg_fraud_amt = actual_frauds['TransactionAmt'].mean() if len(actual_frauds) > 0 else 0
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Total Fraud Count", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate:.1%}")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered_df['RiskTier'].value_counts().reset_index()
        tier_counts.columns = ['RiskTier', 'Count']
        
        color_map = {'Critical Risk': '#ef4444', 'Suspicious': '#f59e0b', 'Clear': '#10b981'}
        
        fig = px.pie(tier_counts, names='RiskTier', values='Count', hole=0.5,
                     color='RiskTier', color_discrete_map=color_map)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_chart2:
        st.subheader("Fraud Rate by Hour of Day")
        if 'HourOfDay' in filtered_df.columns:
            hourly = filtered_df.groupby('HourOfDay').agg(
                total=('isFraud', 'count'),
                fraud=('isFraud', 'sum')
            ).reset_index()
            hourly['FraudRate'] = hourly['fraud'] / hourly['total']
            
            fig2 = px.bar(hourly, x='HourOfDay', y='FraudRate', 
                          labels={'FraudRate': 'Fraud Rate', 'HourOfDay': 'Hour of Day'})
            fig2.update_layout(yaxis_tickformat='.1%', margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("HourOfDay feature not available in dataset.")

elif page == "Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    
    # Allow filtering by exact ID
    search_id = st.text_input("Search by TransactionID (exact match):", "")
    
    display_df = filtered_df.copy()
    if search_id:
        try:
            search_id = int(search_id)
            display_df = display_df[display_df['TransactionID'] == search_id]
        except ValueError:
            st.warning("TransactionID must be a number.")
            
    # Select columns to display
    cols_to_show = ['TransactionID', 'TransactionAmt', 'isFraud', 'FraudProbability', 'RiskTier', 'PredictedFraud']
    if 'HourOfDay' in display_df.columns: cols_to_show.append('HourOfDay')
    if 'ProductCD' in display_df.columns: cols_to_show.append('ProductCD')
    
    st.dataframe(
        display_df[cols_to_show].sort_values('FraudProbability', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "TransactionID": st.column_config.NumberColumn(
                "TransactionID",
                format="%d"
            ),
            "FraudProbability": st.column_config.ProgressColumn(
                "Risk Score",
                help="Model predicted probability of fraud",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "TransactionAmt": st.column_config.NumberColumn(
                "Amount ($)",
                format="$%.2f"
            ),
            "RiskTier": st.column_config.TextColumn(
                "Risk Tier"
            )
        }
    )

elif page == "SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    st.markdown("Enter a TransactionID to understand **why** the model assigned its risk score.")
    
    # Initialize explainer
    @st.cache_resource
    def get_explainer():
        model = artifacts['model']
        # For LightGBM, we can use TreeExplainer directly
        return shap.TreeExplainer(model)
        
    explainer = get_explainer()
    
    tx_id_input = st.text_input("TransactionID:", "")
    
    if tx_id_input:
        try:
            tx_id = int(tx_id_input)
            tx_data = df[df['TransactionID'] == tx_id]
            
            if len(tx_data) == 0:
                st.error(f"TransactionID {tx_id} not found in the sample dataset.")
            else:
                idx = tx_data.index[0]
                row_scaled = X_scaled[idx:idx+1]
                
                # Compute SHAP values
                shap_values = explainer(row_scaled)
                
                # Create visual layout
                col_info, col_shap = st.columns([1, 2])
                
                with col_info:
                    prob = tx_data['FraudProbability'].values[0]
                    tier = tx_data['RiskTier'].values[0]
                    amt = tx_data['TransactionAmt'].values[0] if 'TransactionAmt' in tx_data.columns else 0
                    is_fraud = tx_data['isFraud'].values[0]
                    
                    st.subheader("Transaction Details")
                    st.write(f"**Amount:** ${amt:.2f}")
                    st.write(f"**True Label:** {'Fraud 🔴' if is_fraud else 'Legitimate 🟢'}")
                    
                    if tier == 'Critical Risk': color = 'red'
                    elif tier == 'Suspicious': color = 'orange'
                    else: color = 'green'
                        
                    st.markdown(f"### Risk Score: <span style='color:{color}'>{prob:.1%}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Tier:** {tier}")
                    
                    # Generate Plain English Explanation
                    st.subheader("Plain-English Explanation")
                    
                    # Extract top features
                    sv = shap_values.values[0]
                    feature_names = artifacts['features']
                    
                    # Sort features by absolute impact
                    impacts = [(feature_names[i], sv[i], tx_data[feature_names[i]].values[0]) for i in range(len(sv))]
                    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    top_positive = [i for i in impacts if i[1] > 0][:3]
                    top_negative = [i for i in impacts if i[1] < 0][:3]
                    
                    if prob >= artifacts['threshold']:
                        st.error("This transaction was flagged as **High Risk**.")
                        st.write("Top factors increasing fraud risk:")
                        for fname, shp, val in top_positive:
                            st.write(f"- **{fname}** (value: {val:.2f}) increased the risk score.")
                    else:
                        st.success("This transaction was flagged as **Low Risk**.")
                        st.write("Top factors decreasing fraud risk:")
                        for fname, shp, val in top_negative:
                            st.write(f"- **{fname}** (value: {val:.2f}) lowered the risk score.")
                            
                with col_shap:
                    st.subheader("SHAP Waterfall Plot")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
                    
        except ValueError:
            st.warning("Please enter a valid numeric TransactionID.")
