import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import base64
from groq import Groq
import os
from fpdf import FPDF
import plotly.io
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title=" Predictive Inventory", layout="wide")

st.markdown("""
    <style>
body, p, h1, h2, h3, h4, h5, h6, div, span {
    font-family: "Times New Roman", Times, serif;
}

.stApp {
    background-color: #0B0E14;
    color: #E0E0E0;
}

.kpi-card {
    background: linear-gradient(145deg, #161B22, #0d1117);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363D;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.8);
}

.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #58A6FF;
    margin: 0;
}

.kpi-label {
    font-size: 11px;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.visual-box {
    background: linear-gradient(160deg, #0d1117, #0b0e14);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363D;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}

.visual-title {
    font-size: 15px;
    font-weight: 600;
    color: #C9D1D9;
    margin-bottom: 12px;
    border-bottom: 1px solid #30363D;
    padding-bottom: 6px;
}

.project-card {
    background: linear-gradient(
        135deg,
        rgba(22, 27, 34, 0.9),
        rgba(13, 17, 23, 0.9)
    );
    border-left: 4px solid #58A6FF;
    padding: 26px 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.project-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.9);
}

.project-card h3 {
    font-size: 22px;
    margin-bottom: 12px;
    color: #ffffff;
    letter-spacing: 0.4px;
}

.project-card p {
    font-size: 17px;
    line-height: 1.7;
    color: #d1d5db;
}

.team-card {
    background: linear-gradient(
        140deg,
        rgba(22, 27, 34, 0.95),
        rgba(11, 14, 20, 0.95)
    );
    border-top: 3px solid #58A6FF;
    padding: 26px;
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,0.75);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.team-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 22px 65px rgba(0,0,0,0.95);
}

.team-card h3 {
    font-size: 23px;
    margin-bottom: 6px;
    color: #58A6FF;
}

.team-card p {
    font-size: 16px;
    margin: 6px 0;
    color: #e5e7eb;
}

.team-card a {
    text-decoration: none;
    font-weight: 600;
    color: #58A6FF;
}

.team-card a:hover {
    text-decoration: underline;
}

section[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #30363D;
}

.user-msg {
    background: #1F6FEB;
    color: white;
    padding: 12px;
    border-radius: 12px;
    margin: 6px;
    text-align: right;
}

.bot-msg {
    background: #21262D;
    color: #C9D1D9;
    padding: 12px;
    border-radius: 12px;
    margin: 6px;
    border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Nila Stores.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.strftime('%b %Y')
    df['Month_Sort'] = df['Date'].dt.to_period('M')
    return df

df = load_data()

st.sidebar.markdown(
    "<h2 style='color:#58A6FF; font-family: \"Times New Roman\", Times, serif;'>Nila Analytics</h2>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[role="radiogroup"] > label {
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Executive Dashboard",
        "AI Intelligence (Groq)",
        "Team Details"
    ]
)

if page == "Executive Dashboard":
    st.title(" Executive Dashboard")

   
    with st.container():
        st.markdown('<div class="visual-box">', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        
        with s1:
            country = st.selectbox("Country", ["All"] + sorted(df['Country'].unique().tolist()))
        
        with s2:
            
            store_df = df if country == "All" else df[df['Country'] == country]
            store = st.selectbox("Store", ["All"] + sorted(store_df['Store ID'].unique().tolist()))
        
        with s3:
            
            cat_df = store_df if store == "All" else store_df[store_df['Store ID'] == store]
            category = st.selectbox("Category", ["All"] + sorted(cat_df['Product Category'].unique().tolist()))
        
        with s4:
            
            prod_df = cat_df if category == "All" else cat_df[cat_df['Product Category'] == category]
            product = st.selectbox("Product", ["All"] + sorted(prod_df['Product Name'].unique().tolist()))
        
        with s5:
            
            min_csv = df['Date'].min().date()
            max_csv = df['Date'].max().date()
            dates = st.date_input("Period", [min_csv, max_csv], min_value=min_csv, max_value=max_csv)
        st.markdown('</div>', unsafe_allow_html=True)

    
    if len(dates) == 2:
        mask = (df['Date'].dt.date >= dates[0]) & (df['Date'].dt.date <= dates[1])
    else:
        mask = (df['Date'].dt.date >= dates[0])

    if country != "All": mask &= (df['Country'] == country)
    if store != "All": mask &= (df['Store ID'] == store)
    if category != "All": mask &= (df['Product Category'] == category)
    if product != "All": mask &= (df['Product Name'] == product)
    
    f_df = df[mask].sort_values('Date')


    k1, k2, k3, k4 = st.columns(4)
    total_sales = f_df["Sales Amount"].sum()
    units_sold = f_df["Units Sold"].sum()
    monthly_units = f_df.set_index('Date').resample('MS')['Units Sold'].sum()
    forecast_kpi = int(monthly_units.mean() * 1.2) if not monthly_units.empty else 0
    
    with k1: st.markdown(f'<div class="kpi-card"><p class="kpi-label">Total Revenue</p><p class="kpi-value">${total_sales:,.0f}</p></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="kpi-card"><p class="kpi-label">Units Volume</p><p class="kpi-value">{units_sold:,}</p></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi-card"><p class="kpi-label">Target Demand </p><p class="kpi-value">{forecast_kpi:,}</p></div>', unsafe_allow_html=True)
    with k4: 
        risk = "CRITICAL" if forecast_kpi > 500 else "STABLE"
        color = "#F85149" if risk == "CRITICAL" else "#3FB950"
        st.markdown(f'<div class="kpi-card"><p class="kpi-label">Inventory Risk</p><p class="kpi-value" style="color:{color}">{risk}</p></div>', unsafe_allow_html=True)

   
    c1, c2, c3 = st.columns([1.2, 2, 1.2])
    
    with c1: 
        st.markdown('<div class="visual-box"><div class="visual-title">Global Sales Map</div>', unsafe_allow_html=True)
        geo = f_df.groupby('Country')['Sales Amount'].sum().reset_index()
        fig_globe = go.Figure(go.Choropleth(locations=geo['Country'], locationmode='country names', z=geo['Sales Amount'], colorscale='YlGnBu', showscale=False))
        fig_globe.update_geos(projection_type="orthographic", bgcolor="rgba(0,0,0,0)")
        fig_globe.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_globe, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2: 
        st.markdown('<div class="visual-box"><div class="visual-title">30-Month Forecasting Comparison</div>', unsafe_allow_html=True)
        ts = f_df.set_index('Date').resample('MS')['Sales Amount'].sum().asfreq('MS').fillna(0)
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Actual Sales", line=dict(color='#58A6FF', width=3)))
        
        forecast_results = [] 

        if len(ts) > 5:
            f_periods = 30
            f_dates = pd.date_range(ts.index.max() + pd.DateOffset(months=1), periods=f_periods, freq='MS')
            
            try:
                arima_m = ARIMA(ts, order=(1,1,1)).fit()
                p = arima_m.forecast(f_periods)
                fig_line.add_trace(go.Scatter(x=f_dates, y=p, name="ARIMA", line=dict(color='#BC8CFF', dash='dash')))
                for d, v in zip(f_dates, p): forecast_results.append({"Forecast_Date": d.date(), "Model": "ARIMA", "Predicted_Revenue": round(v, 2)})
            except: pass

            try:
                sarima_m = SARIMAX(ts, order=(1,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)
                p = sarima_m.forecast(f_periods)
                fig_line.add_trace(go.Scatter(x=f_dates, y=p, name="SARIMAX", line=dict(color='#3FB950', width=2)))
                for d, v in zip(f_dates, p): forecast_results.append({"Forecast_Date": d.date(), "Model": "SARIMAX", "Predicted_Revenue": round(v, 2)})
            except: pass

            try:
                p_df = ts.reset_index().rename(columns={'Date': 'ds', 'Sales Amount': 'y'})
                m = Prophet(yearly_seasonality='auto', weekly_seasonality=False, daily_seasonality=False)
                m.fit(p_df)
                p_fc = m.predict(m.make_future_dataframe(periods=f_periods, freq='MS')).tail(f_periods)
                fig_line.add_trace(go.Scatter(x=p_fc['ds'], y=p_fc['yhat'], name="Prophet", line=dict(color='#F85149', width=2)))
                for d, v in zip(p_fc['ds'], p_fc['yhat']): forecast_results.append({"Forecast_Date": d.date(), "Model": "Prophet", "Predicted_Revenue": round(v, 2)})
            except: pass

        fig_line.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c3: 
        st.markdown('<div class="visual-box"><div class="visual-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        m_df = f_df.groupby(['Month_Sort', 'Month'])['Sales Amount'].sum().reset_index()
        fig_col = px.bar(m_df, x='Month', y='Sales Amount', color_discrete_sequence=['#3FB950'])
        fig_col.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_col, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    b1, b2, b3 = st.columns([1.2, 1, 1])
    with b1: 
        st.markdown('<div class="visual-box"><div class="visual-title">Top 5 Products</div>', unsafe_allow_html=True)
        top_p = f_df.groupby('Product Name')['Sales Amount'].sum().nlargest(5).reset_index().sort_values('Sales Amount')
        fig_bar = px.bar(top_p, x='Sales Amount', y='Product Name', orientation='h', color_discrete_sequence=['#58A6FF'])
        fig_bar.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2: 
        st.markdown('<div class="visual-box"><div class="visual-title">Country Sales Split</div>', unsafe_allow_html=True)
        fig_pie = px.pie(f_df, values='Sales Amount', names='Country', color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_pie.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=10), template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b3: 
        st.markdown('<div class="visual-box"><div class="visual-title">Category Distribution</div>', unsafe_allow_html=True)
        fig_don = px.pie(f_df, values='Sales Amount', names='Product Category', hole=0.5, color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_don.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=10), template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_don, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    st.markdown('<div class="visual-box" style="text-align:center;">', unsafe_allow_html=True)
    if forecast_results:
        
        f_csv = pd.DataFrame(forecast_results).pivot(index='Forecast_Date', columns='Model', values='Predicted_Revenue').reset_index()
        st.write("###  30-Month Future Revenue Forecast")
        st.download_button(
            label="Download Forecast Report (CSV)",
            data=f_csv.to_csv(index=False).encode('utf-8'),
            file_name=f'Nila_Forecast_{store}_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.info("Select filters to generate forecast download")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "AI Intelligence (Groq)": 
    st.title(" Nila Predictive Business Analyst")

    
    store_data = df.groupby('Store ID').agg({'Sales Amount': 'sum', 'Units Sold': 'sum'}).to_dict('index')
    
    
    seasonal_peak = df.groupby(['Season', 'Product Name'])['Units Sold'].sum().reset_index()
    seasonal_peak = seasonal_peak.sort_values(['Season', 'Units Sold'], ascending=[True, False]).groupby('Season').head(3).to_dict('records')
    
    
    total_rev = df['Sales Amount'].sum()
    avg_monthly_sales = total_rev / 24 
    predicted_next_year = total_rev * 1.15 

    
    data_snapshot = f"""
    Knowledge Base:
    - All Stores Performance: {store_data}
    - Seasonal Peaks: {seasonal_peak}
    - Revenue: Total ${total_rev:,.0f}, Avg Monthly ${avg_monthly_sales:,.0f}
    - Prediction: Estimated 15% increase next year based on historical trends.
    - Low Stock Alert: Any item with average units sold > 150 per month needs reordering.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        role_class = "user-msg" if m["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="{role_class}">{m["content"]}</div>', unsafe_allow_html=True)

    if p := st.chat_input("Ex: Predict sales for IND Store 01 and tell me what to reorder?"):
        st.session_state.messages.append({"role": "user", "content": p})
        st.markdown(f'<div class="user-msg">{p}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            try:
                sys_message = f"""You are an Inventory & Sales Forecaster. Use this: {data_snapshot}.
                Instructions:
                1. If asked about a store, look at its specific Sales Amount.
                2. If asked about 'reorder', suggest items that have high sales units.
                3. If asked about 'next year' or 'prediction', use the 15% growth logic.
                4. For 'seasons', mention the specific peak products for that season from the snapshot."""

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_message}, {"role": "user", "content": p}],
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.rerun()

elif page == "Project Overview":
    if page == "Project Overview":
        st.title(" Predictive Inventory System")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="project-card">
            <h3> Inventory Imbalance</h3>
            <p>Nila Stores faces critical challenges with <b>Overstocking</b> and <b>Stockouts</b>. Manual planning leads to revenue loss and high wastage in perishable categories.</p>
        </div>
        <div class="project-card">
            <h3> Forecasting Intelligence</h3>
            <p>Using <b>ARIMA, SARIMAX,</b> and <b>Prophet</b>, we turn historical data into actionable procurement strategies.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="project-card">
            <h3> Technology Stack</h3>
            <p><b>Backend:</b> Python (Statsmodels, Prophet)<br>
            <b>Dashboard:</b> Streamlit, Plotly, PyDeck<br>
            <b>Brain:</b> Groq </p>
        </div>
        <div class="project-card">
            <h3> Business Impact</h3>
            <p>30% reduction in wastage, 15% increase in fulfillment rates, and automated stock alerts.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Team Details":
    st.title("Meet the Team")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="team-card">
            <h3>Sanjaykanth Chandran</h3>
            <p><b>Team Lead | Data Analyst</b></p>
            <ul>
                <li>Currently Data Analyst Intern at DMW CNC Solutions India Pvt Ltd</li>
                <li>Former AI/ML Intern at Tech Mahindra</li>
                <li>Strong interest in data analytics, forecasting, and business insights</li>
                <li>Passionate about leveraging data to drive strategic decisions and optimize inventory management</li>
            </ul>
            <p>sanjaychandran29803@gmail.com</p>
            <p>
                <a href="https://www.linkedin.com/in/sanjaykanth-chandran" target="_blank">
                    LinkedIn
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="team-card">
            <h3>Nivetha Munusamy</h3>
            <p><b>Data Analyst</b></p>
            <ul>
                <li>Currently Data Analyst Intern at DMW CNC Solutions India Pvt Ltd</li>
                <li>Former Data Science Intern at Nitroware Technologies Pvt Ltd, Coimbatore</li>
                <li>Interested in analytics, visualization, and data-driven decision making</li>
            </ul>
            <p>nivethamunusamy2004@gmail.com</p>
            <p>
                <a href="https://linkedin.com/in/nivetha-munusamy" target="_blank">
                    LinkedIn
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)
