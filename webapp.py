
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

st.markdown('##')
st.markdown('##')
st.image('images/Are_you_the_one.jpeg')
st.markdown('')

st.write("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Predicting druggability of protein targets</h1>", unsafe_allow_html=True)


###########################################
# Add a sidebar with content:
st.sidebar.header('Final Project Presentation')
st.sidebar.image('images/spiced.png', width=200)
st.sidebar.markdown('''
    Data Science Bootcamp    
    Lisa Kalfhues      
    _19.05.2023_
    ''')

st.sidebar.write('##')
st.sidebar.header('Code availability')
st.sidebar.write('The code for this project is available under the [GNU License](https://www.gnu.org/licenses/gpl-3.0) in this [GitHub repository](https://github.com/LisaKalfhues/Druggability_Predictor/tree/main)')
st.sidebar.write('##')
st.sidebar.header('Thank you!')
st.sidebar.write('Thanks to Spiced, all the amazing teachers and my Nigelas 🫶')

###########################################
