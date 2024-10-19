import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, auc




st.set_page_config(page_title='Diabetes Prediction', layout='wide', initial_sidebar_state='expanded')

# load the pickle file
default_model = pickle.load(open('pickle_files/rfc_model.pkl', 'rb'))
default_scaler = pickle.load(open('pickle_files/scaler.pkl', 'rb'))


# user input from client
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 3)
    glucose = st.sidebar.slider('Glucose', 0, 300, 100)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 200, 80)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 1000, 100)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    age = st.sidebar.slider('Age', 0, 100, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# display the user input
st.subheader('👤 User Input parameters')
st.write(input_df)

# preprocess the user input
input_df_scaled = default_scaler.transform(input_df)

# make prediction
prediction = default_model.predict(input_df_scaled)[0]


# display the prediction
if prediction == 0:
    st.subheader('Non-Diabetic')
else:
    st.subheader('Diabetic')
