import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# page main title
st.set_page_config(page_title="Stock Prediction", layout="wide")

# App - Main heading
st.markdown("<h1 style='text-align: center; color: #FFD700;'>Stock Price Prediction App</h1>", unsafe_allow_html=True)

# Sidebar for selecting
option = st.sidebar.selectbox("Select a Stock", ["Tesla", "Reliance", "AAPL"])

# For title based on selection
if option == "Tesla":
    title_text = "Tesla Stock Prediction"
    data_path = r"D:\My Data\Downloads\TSLA.csv"
elif option == "Reliance":
    title_text = "Reliance Stock Prediction"
    data_path = r"D:\My Data\Downloads\RELIANCE.NS.csv"
elif option == "AAPL":
    title_text = "AAPL Stock Prediction"
    data_path = r"D:\My Data\Downloads\AAPL.csv"

# Display title style
st.markdown(f"<h2 style='text-align: center; color: #ff0078d7;'>{title_text}</h2>", unsafe_allow_html=True)

# Load dataset path
data = pd.read_csv(data_path)

# Date column to datetime format covertion 
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

# Handle missing values for nuber coloumn
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# having only number coloumn
X = data[['High', 'Low', 'Open', 'Volume']].copy()
y = data['Close'].copy()

# Splitting into Train-Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# apply Linear Regression Model to split
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Layout defining 
col1, col2 = st.columns([2, 1])
#left side
with col1:
    st.markdown("<h3>Stock Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(data.head(10))
#right side
with col2:
    st.markdown("<h3>Model Training Info</h3>", unsafe_allow_html=True)
    st.write(f"**Selected Stock:** {option}")
    st.write(f"**Number of Entries:** {data.shape[0]}")
    st.write(f"**Features Used:** High, Low, Open, Volume")
    st.write(f"**Model Type:** Linear Regression")

# predicting the data from user input 
st.markdown("<h2>Predict the Closing Price</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    high = st.number_input("High Price:", min_value=0.0, format="%.2f")
with col2:
    low = st.number_input("Low Price:", min_value=0.0, format="%.2f")
with col3:
    open_price = st.number_input("Open Price:", min_value=0.0, format="%.2f")
with col4:
    volume = st.number_input("Volume:", min_value=0, format="%d")

if st.button("Predict Closing Price"):
    input_data = np.array([[high, low, open_price, volume]])
    predicted_price = regressor.predict(input_data)
    st.success(f"Predicted Closing Price for {option}: **{predicted_price[0]:.2f}**")
#############################################################################################################################################################
