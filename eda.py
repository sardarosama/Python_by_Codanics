import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas_profiling as pp
from streamlit_pandas_profiling import st_profile_report

# Title of Web App

st.markdown('''
# **Exploratory Data Analysis Web Application**
This app is developed by Osama Shakeel to help you show exploratory 
data analysis on your data named as "EDA App"
''')
# how to upload a file from your computer
st.header("Upload your dataset (.csv)")
with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    df = sns.load_dataset("titanic")
    st.sidebar.markdown("[Example csv file](df)")



