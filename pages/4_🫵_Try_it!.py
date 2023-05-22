import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import matplotlib.ticker as ticker
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

###########################################
# Add a sidebar with content:

st.sidebar.header('Final Project Presentation')
st.sidebar.image('images/spiced.png', width=200)
st.sidebar.markdown('''
    Data Science Bootcamp    
    Lisa Kalfhues      
    _19.05.2023_
    ''')
###########################################

###########################################
# Add presentation for "Try it":

st.image('images/try.jpg')
st.markdown('')

import joblib

# load model
model = joblib.load("Model/model_predicting_druggability.pkl")

# create DataFrames ffor choices
df_protein1 = pd.read_csv("Model/df_protein1.csv")
df_protein2 = pd.read_csv("Model/df_protein2.csv")

# Titel and text for webapp
st.markdown("<h2 style='text-align: center;'>Protein druggability predictor</h2>", unsafe_allow_html=True)
st.markdown('##')
st.markdown('##')

# Dropdown menu zur Auswahl der Protein-ID
protein_choices = ['','T81850', 'T85943']
protein_text = st.subheader('Please select a protein ID:')
protein_choice = st.selectbox('', protein_choices, index=0)

if protein_choice != '':
    
    if protein_choice == 'T81850':
        protein_file = 'Model/df_protein1.csv'
    elif protein_choice == 'T85943':
        protein_file = 'Model/df_protein2.csv'

    df_protein = pd.read_csv(protein_file)

# Show selcted protein features
    st.markdown('##')
    st.markdown('##')
    st.subheader(f"Properties of the protein ({protein_choice}):")


# first container
    col1, col2 = st.columns(2)
    col1.markdown("<h4 style='text-align: left;'>KEGG pathway</h4>", unsafe_allow_html=True)
    col1.text(df_protein.iloc[0,0])
    col2.markdown("<h4 style='text-align: left;'>Biochemical class</h4>", unsafe_allow_html=True)
    col2.text(df_protein.iloc[0, 1])

# second container
    col3, col4, col5 = st.columns(3)
    col3.markdown("<h4 style='text-align: left;'>Enzyme class</h4>", unsafe_allow_html=True)
    col3.text(df_protein.iloc[0, 2])
    col4.markdown("<h4 style='text-align: left;'>3D Structure</h4>", unsafe_allow_html=True)
    col4.text(df_protein.iloc[0, 3])
    col5.markdown("<h4 style='text-align: left;'>Mode of action</h4>", unsafe_allow_html=True)
    col5.text(df_protein.iloc[0, 4])



st.write('<style>div.row-widget.stButton>button{background-color: #F4F3EC; color: #B318AE; font-size: 44px;}</style>', unsafe_allow_html=True)
if st.button('Are you the one?', use_container_width=True):
    X = df_protein
    
    # predict druggability
    prediction = model.predict(X)[0]
    
    # show prediction
    if prediction == 0:
        st.error(f"The selected protein ({protein_choice}) is **not** druggable.", icon='❌')
    else:
        st.success(f"The selected protein ({protein_choice}) is druggable.", icon="✅")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')


import webbrowser
st.write('<style>div.row-widget.stButton>button{background-color: #F4F3EC; color: #181816; font-size: 44px;}</style>', unsafe_allow_html=True)
if st.button("Check your prediction here", use_container_width=True):
    if protein_choice == 'T81850':
        url = 'https://db.idrblab.net/ttd/search/ttd/target?search_api_fulltext=T81850'
    elif protein_choice == 'T85943':
        url = 'https://db.idrblab.net/ttd/search/ttd/target?search_api_fulltext=T85943'
    webbrowser.open_new_tab(url)

#############################################