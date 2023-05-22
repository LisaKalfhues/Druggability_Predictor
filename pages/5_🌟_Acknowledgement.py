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
# Add presentation for "Thanks":

st.image('images/thankyou.jpg')
st.markdown("<h3 style='text-align: center;'>Thank you Spiced and my Nigelas 🫶</h3>", unsafe_allow_html=True)
st.image('images/nigelas.png')

###########################################

###########################################
# Add presentation for "Tools":

st.markdown("<h3 style='text-align: center;'>What tools did I use?</h3>", unsafe_allow_html=True)
col1, col2, col3,col4 = st.columns(4)
col1.image('images/python.png')
col2.image('images/docker.jpg')
col3.image('images/streamlit.png')
col4.image('images/sklearn.png')

col5, col6, col7, col8 = st.columns(4)
col5.image('images/seaborn.png')
col6.image('images/numpy.png')
col7.image('images/pandas.png')
col8.image('images/matplotlib.png')

###########################################

