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
# Add presentation for "Data":
st.image('images/data.jpg')

st.write("##")
st.write("##")
st.markdown("<h2 style='text-align: center;'>Data sources:</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Public databases providing data on proteins</h4>", unsafe_allow_html=True)
st.write("##")
col4, col5 = st.columns(2, gap='small')
col4.image('images/TTD_logo.png', caption = 'The Therapeutic Target Database')
col5.image('images/PBD_world.jpg', caption = 'Protein Data Bank')
st.write("##")
col6, col7 = st.columns(2, gap='small')
col6.image('images/SCOP.jpg', caption = 'SCOP: Protein Structure Classification Database')
col7.image('images/uniprot_logo.png', caption = 'Universal Protein resource')
st.write("##")
st.write("##")

st.markdown("<h2 style='text-align: center;'>Exploratory data analysis</h2>", unsafe_allow_html=True)
st.write("##")

st.markdown("<h4 style='text-align: center;'>Protein properties:</h4>", unsafe_allow_html=True)
txt =st.text('''
    KEGG pathway    Biochemical class    Enzyme class    Structure ID    Mode of action
    ''')
st.write("##")
st.write("##")

text_center_style = '''
    display: flex;
    justify-content: center;
'''

col3, col4 = st.columns(2)
with col3:
    st.markdown(f'<p style="{text_center_style}">✅ Successful targets</p>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>1688</h2>", unsafe_allow_html=True)
with col4:
    st.markdown(f'<p style="{text_center_style}">❌ Failed targets</p>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>85</h2>", unsafe_allow_html=True)

###########################################