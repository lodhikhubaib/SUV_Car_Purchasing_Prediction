import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
suv_car_df = pd.read_csv('suv_data.csv')

# Split data into train and test sets
X = suv_car_df.iloc[:,[2,3]]
y = suv_car_df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

def app():
    image_path = 'car.jpeg'
    with st.container():
        st.image(image_path,width = 350)
    st.title("SUV Car Purchasing Prediction")
    st.write("This App predict whether a customer will purchase an SUV car based in their age and Salary")

    age = st.slider("Select Age: ", min_value = 18,max_value=100,step = 1,value =30)
    salary = st.slider('Select Salary: ',min_value=10000,max_value=2000000,step=1000,value=50000)

    x_new = [[age,salary]]
    x_new_scaled = sc.transform(x_new)
    y_new = model.predict(x_new_scaled)

    if y_new == 1:
        st.write("This Person has bought the SUV Car")
    else:
        st.write("This Person has not bought the SUV Car")

if __name__ == '__main__':
    app()