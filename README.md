Nila Analytics: Predictive Inventory System

Nila Analytics is a comprehensive data science application designed to mitigate inventory imbalances specifically overstocking and stockouts. By integrating statistical time-series models and a Large Language Model (LLM), this platform converts historical retail data into optimized procurement strategies.

Project Overview
Nila Stores faces significant challenges with manual inventory planning, which often results in revenue loss and high wastage in perishable categories. This system provides an automated solution using three primary pillars: Data Visualization, Advanced Forecasting, and Artificial Intelligence.

Core Functionalities
1. Executive Dashboard
The dashboard provides a high level overview of business health with Power BI-style interactivity.

Dynamic Filtering: Users can slice data by Country, Store ID, Product Category, and specific time periods.

KPI Metrics: Real time tracking of Total Revenue, Unit Volume, and Target Demand.

Inventory Risk Assessment: An automated logic engine that flags stock levels as "Critical" or "Stable" based on projected demand spikes.

2. Multi-Model Forecasting Engine
To ensure high accuracy, the system calculates and compares three different forecasting methodologies:

ARIMA (AutoRegressive Integrated Moving Average): Best for capturing linear trends.

SARIMAX: Extends ARIMA by accounting for seasonal patterns and external variables.

Prophet: A robust model developed by Meta, specifically tuned for handling outliers, missing data, and holiday effects in retail.

3. AI Intelligence (Ollama Integration)
The application features a "Data Aware" assistant powered by the Gemma 3 (4b) model.

Contextual RAG: The assistant does not just provide general information; it queries the underlying CSV data to answer specific questions about your sales, top performing products, and regional revenue.

Privacy Focused: The LLM runs locally via Ollama, ensuring that sensitive business data is never sent to external cloud servers.

Technical Stack
User Interface: Streamlit

Data Manipulation: Pandas, NumPy

Visualization: Plotly (Express, Graph Objects, and Geospatial Maps)

Forecasting Models: Statsmodels (ARIMA/SARIMAX), Prophet

AI Engine: Ollama (Gemma 3:4b)

PDF Reporting: FPDF

Installation and Usage

Prerequisites

Python 3.9 or higher

Ollama installed locally

Development Team


Sanjaykanth Chandran: Team Lead and Data Analyst. Specialized in forecasting and business insights.

Nivetha Munusamy: Data Analyst. Specialized in data visualization and statistical decision making.
