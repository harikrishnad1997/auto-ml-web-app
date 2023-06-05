from operator import index
import streamlit as st
import plotly.express as px
import numpy as np
from pycaret.regression import setup, compare_models, pull, save_model, load_model, plot_model
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://michael-fuchs-python.netlify.app/post/2022-01-01-automl-using-pycaret-classification_files/p133s1.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
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
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        print("WIP")
        # h2o.init()
        # df = h2o.import_file(df)
        # df.describe(chunk_summary=True)
        # train, test = df.split_frame(ratios=[0.8], seed = 1)
        # aml = H2OAutoML(max_models =25,
        #         balance_classes=True,
		# seed =16548846)
        # aml.train(training_frame = train, y = 'y')
        # lb = aml.leaderboard
        # lb.head(rows=lb.nrows)
        # aml.train(training_frame = train, y = 'y', leaderboard_frame = my_leaderboard_frame)
        # best_model = aml.get_best_model()
        # model_path = h2o.save_model(model=best_model,force=True)
        setup(df.dropna(subset=chosen_target), target=chosen_target, session_id = 2774764,imputation_type = 'simple',numeric_imputation='mean',categorical_imputation='mode')
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        # plot_model(best_model, plot='residuals', display_format='streamlit')
        # plot_model(best_model, plot='feature', display_format='streamlit')
        # plot_model(best_model, plot='error', display_format='streamlit')
        save_model(best_model, 'best_model')
        # y = df[chosen_target]
        # X = df.loc[:, df.columns!=chosen_target]
        # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =65481254)
        # reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
        # models,predictions = reg.fit(X_train, X_test, y_train, y_test)
        # st.dataframe(models)
        # model_dictionary = reg.provide_models(X_train,X_test,y_train,y_test)

if choice == "Download": 
    print("Working")
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

