from pyexpat import features, model
from tkinter import N
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix

# Make Containers

header = st.container()
data_sets = st.container()
featuress = st.container()
model_training = st.container()

with header:
    st.title("Kashti Dataset app")
    st.text("This is a test")

with data_sets:
    st.title("Kashti Doob gaye")
    st.text("We will work on titanic dataset")

    # Load the data
    st.header("araay ho sambha kitnay admi thay")

    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head(10))
    st.bar_chart(df["sex"].value_counts())

    #Other plots
    st.header("Class ka hissab sa farq")
    st.bar_chart(df["class"].value_counts())

    # age barplot
    st.bar_chart(df['age'].sample(10))



with featuress:
    st.header("These are all the features of the app")
    st.text("Asaan hai itna mushkal nhi hai")
    st.markdown('1. **Features 1** This will tell us something')
    st.markdown('2. **Features 2** This will tell us something')


with model_training:
    st.header("Kashti walon ka kia bana")
    st.text("We will be tranning a model")
    # Making columns
    input, display = st.columns(2)

    # Pehlay columns main ap ka selection points hon
    max_depth = input.slider("how many people do you want to select", min_value=10, max_value=100, value = 20 , step=5)
    
#n_estimators
n_estimators = input.selectbox("how many people do you want to select", options = [50,100,200,300,  'NO LIMIT'])


# Adding list of features
input.write(df.head()) 
input.write(df.columns) 

# input features from users

input_features = input.text_input("which features you would use?")

# Machine learning ka model kasy banain?
model = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)

# Yahan par ak condition lagayegay 
if n_estimators == 'No LIMIT':
    model = RandomForestRegressor(max_depth = max_depth)
else:
    model = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)


# define input and output

X = df[[input_features]]
y = df[['fare']]

# fit a model
model.fit(X,y)
pred = model.predict(y)

# Display Metrices
display.subheader("Mean absolute error of model is ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of model is ")
display.write(mean_squared_error(y, pred))
display.subheader("r square score of model is ")
display.write(r2_score(y, pred))

