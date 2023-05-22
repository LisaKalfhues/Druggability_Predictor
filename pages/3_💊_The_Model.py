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
# Add presentation for "Model":

st.image('images/model.jpg')
        
st.write("##")
st.write("##")
st.markdown("<h3 style='text-align: center;'>The model pipeline:</h3>", unsafe_allow_html=True)
st.image('images/pipeline.jpg')

st.write("##")
st.write("##")
st.markdown("<h3 style='text-align: center;'>Model evaluation:</h3>", unsafe_allow_html=True)
st.image('Model/confusion_matrix.png')

###########################################