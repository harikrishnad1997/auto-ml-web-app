from operator import index
import streamlit as st
import plotly.express as px
import numpy as np
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
from sklearn.datasets import load_iris
import os

@st.cache_resource
def load_data():
    X, y = load_iris(return_X_y=True)
    df = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    df['target'] = y
    return df

df = load_data()

with st.sidebar:
    st.image("https://analyticsindiamag.com/wp-content/uploads/2020/10/lazypredict.jpg")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    # AV = AutoViz_Class()
    # config = {
    #     'filename': None,
    #     'sep': ',',
    #     'depVar': '',
    #     'dfte': df,
    #     'header': 0,
    #     'verbose': 0,
    #     'lowess': False,
    #     'chart_format': 'html',
    #     'max_rows_analyzed': 10000,
    #     'max_cols_analyzed': 50,
    #     # 'save_plot_path': None,
    # }
    # AV.AutoViz()
    # st.components.v1.html(AV.html, width=1000, height=1000, scrolling=True)
    
    AV = AutoViz_Class()
    AV.AutoViz("",dfte=df)
    report_path = os.path.join(os.path.dirname(__file__), "autoviz_report.html")
    try:
        AV.save_html(report_path)
        HtmlFile = open(report_path, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.components.v1.html(source_code, width=1000, height=1000, scrolling=True)
    except:
        st.write("Error occurred while generating the AutoViz report.")


if choice == "Modelling":
    target_options = df.columns.tolist()
    chosen_target = st.selectbox('Choose the Target Column', target_options)
    model_type = st.selectbox('Choose Model Type', ['Regression', 'Classification'])
    if st.button('Run Modelling'):
        X = df.drop(chosen_target, axis=1)
        y = df[chosen_target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=65481254)

        if model_type == 'Classification':
            # Classification
            clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
        else:
            # Regression
            reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)
            model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)

        st.dataframe(models)

if choice == "Download":
    with open('model_dictionary.pkl', 'wb') as f:
        st.download_button('Download Model', f, file_name="model_dictionary.pkl")