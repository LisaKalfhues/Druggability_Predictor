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
# Add presentation for "Introduction":

st.image('images/introduction.jpg')
st.markdown('##')
st.markdown('##')
st.markdown('##')

top_5 = pd.read_csv('Top_5_causes_of_death.csv', sep=',', index_col=0)

ax = sns.lineplot(data=top_5, palette='Paired')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xlabel('Year', fontsize=12)
plt.xlim([2001, 2021])
plt.ylim([0, 1060000])
ax.set_ylabel('Total number of death', fontsize=12)
ax.set_title('Top 5 causes of death', fontsize=14)
lines = ax.get_lines()
for line in lines:
    line.set_linestyle('-')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

fig = ax.get_figure()

st.write(fig)

st.write("""
<style>
.small-grey-center {
    font-size: small;
    color: grey;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.write("<p class='small-grey-center'>Source: Federal Statistic Office. Causes of death (2001-2021).</p>", unsafe_allow_html=True)
st.markdown('##')
st.markdown('##')
st.markdown('##')
st.markdown('##')

st.markdown("<h2 style='text-align: center;'>Why is it so hart to find medications?</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Proteins are the most promising structure to become a drug target</h4>", unsafe_allow_html=True)

col1, col2, col3 =st.columns(3)
with col1:
    st.image('images/protein_all.png')
    st.markdown("<h6 style='text-align: center;'>Proteins are everywhere</h6>", unsafe_allow_html=True)

with col2:
    st.image('images/protein_active.png')
    st.markdown("<h6 style='text-align: center;'>Proteins can get activated</h6>", unsafe_allow_html=True)

with col3:
    st.image('images/protein_signal.png')
    st.markdown("<h6 style='text-align: center;'>Proteins are involved in signaling</h6>", unsafe_allow_html=True)


st.markdown('##')
st.markdown('##')
st.markdown('##')
st.markdown('##')

st.markdown("<h4 style='text-align: center;'>From protein to medication:</h4>", unsafe_allow_html=True)
image = Image.open('images/protein_medication.jpg')
st.image(image, caption='Sun Duxin et al., Why 90% of clinical drug development fails and how to improve it?, Acta Pharmaceutica Sinica B, 2022')

###########################################