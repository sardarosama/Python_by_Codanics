import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas_profiling as pp
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Title of Web App

st.markdown('''
# **Exploratory Data Analysis Web Application**
This app is developed by Osama Shakeel to help you show exploratory 
data analysis on your data named as "EDA App"
''')
# how to upload a file from your computer

with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    df = sns.load_dataset("titanic")
    st.sidebar.markdown("[Example CSV file](https://www.kaggle.com/c/titanic/download/GQf0y8ebHO0C4JXscPPp%2Fversions%2FXkNkvXwqPPVG0Qt3MtQT%2Ffiles%2Ftrain.csv)")


# Profiling reports for pandas dataframe

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(  uploaded_file )
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative = True)
    st.header("input dataframe")
    st.write(df)
    st.write("---")
    st.header("Profiling Report")
    st_profile_report(pr)
else:
    st.info("No file uploaded")
    if st.button("press to use for example data"):
      # example data set
        @st.cache
        def load_data():
            a =pd.DataFrame(np.random.rand(100,5), columns=['a','b','c','d','e'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative = True)
        st.header("input dataframe")
        st.write(df)
        st.write("---")
        st.header("Profiling Report")
        st_profile_report(pr)
        