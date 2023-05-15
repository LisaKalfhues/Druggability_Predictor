
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

st.image('../Are_you_the_one.jpeg')
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
with st.sidebar:
    st.title('Are you the one?')
    st.header('Content')
    st.selectbox(
        'Where would you like to go',
        ('Intro', 'Data', 'Model', 'Try it', 'Tools', 'Thanks')
    )

st.sidebar.header('Code availability')
# LINKS ANPASSEN
st.sidebar.write('The code for this project is available under the [MIT License](https://mit-license.org/) in this [GitHub repo](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp)')

st.sidebar.header('Thank you!')
st.sidebar.write('Thanks to Spiced, all the amazing teachers and my Nigelas 🫶')

###########################################
# Add presentation for "Introduction":
st.subheader ('Why is this important?')

top_5 = pd.read_csv('Top_5_causes_of_death_T_int.csv', sep=',', index_col=0)

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

st.write(fig, caption= '----quelle hiiiier source---')

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
st.markdown('')
st.markdown('')
st.markdown('')

st.subheader('Why is it so hart to find a medication?')
st.write('Proteins are the most promising structure to become a drug target')
col1, col2, col3 =st.columns(3)
col1.image('../protein_all.jpg', caption = 'Proteins are everywhere')
col2.image('../protein_active.jpg', caption = 'Proteins can get activated')
col3.image('../protein_signal.jpg', caption = 'Proteins are involved in signaling')

st.write('From protein to medication:')
image = Image.open('../protein_medication.jpg')
st.image(image, caption='Sun Duxin et al., Why 90% of clinical drug development fails and how to improve it?, Acta Pharmaceutica Sinica B, 2022')


###########################################
# Add presentation for "Data":
st.header('How can AI help us here?')
st.image('../ML_model.jpg')


st.subheader('Where did I get the data from?')

# Use columns
col4, col5, col6, col7 = st.columns(4)
col4.text('TTP Database')
col5.text('PDB world and PDB')
col6.text('Scope')
col7.text('Uniprot')
# Logo hinzufügen
# Link hinzufügen
# Wenn man mit der Maus über das bild fährt einen ausschnit aus den daten zeigen


###########################################
# Add presentation for "Model":
st.subheader('What machine learning model did I use?')
st.code('''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline ''')

col8, col9 =st.columns(2)
col8.code('''
categorical_features = ['KEGG_pathway', 'Biochemical_class', 'Enzyme_class', 'PDB_structure', 'Mode_of_action']
categorical_transformer = OneHotEncoder(sparse=False,handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('OHE', categorical_transformer, categorical_features)
    ]
)

models = [
          ('logreg', LogisticRegression()),
          ('forest', RandomForestClassifier(n_estimators=120))
]
m = VotingClassifier(models)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ('smote', SMOTENC(random_state=11, categorical_features=[0,1,2,3,4,])),
        ('models', m)
    ]
)'''
)
col9.image('../ML_model/pipeline.png')

st.image('../ML_model/confusion_matrix.png')


###########################################
# Add presentation for "Try it":
st.image('../Lets_try_it.jpg')

import joblib

# Laden des trainierten Modells
model = joblib.load("../ML_model/model_predicting_druggability.pkl")

# Erstellen von DataFrames für die verschiedenen Auswahlmöglichkeiten
df_protein1 = pd.read_csv("../ML_model/df_protein1.csv")
df_protein2 = pd.read_csv("../ML_model/df_protein2.csv")

# Titel und Text für die Streamlit-Seite
st.title("Protein Druggability Predictor")

# Dropdown-Menü zur Auswahl der Protein-ID
protein_choices = ['T81850', 'T85943']
protein_choice = st.selectbox('Please select a protein ID:', protein_choices)

# Laden des ausgewählten Protein-Datensatzes
if protein_choice == 'T81850':
    protein_file = '../ML_model/df_protein1.csv'
elif protein_choice == 'T85943':
    protein_file = '../ML_model/df_protein2.csv'

df_protein = pd.read_csv(protein_file)

# Anzeigen der ausgewählten Protein-Features
st.header(f"Properties of the protein ({protein_choice}):")


# Erster Container für Spalten 1 und 2
col1, col2 = st.columns(2)
col1.subheader('KEGG pathway:')
col1.write(df_protein.iloc[0,0])
col2.subheader('Biochemical class:')
col2.write(df_protein.iloc[0, 1])

# Zweiter Container für Spalten 3-5
col3, col4, col5 = st.columns(3)
col3.subheader('Enzyme class:')
col3.write(df_protein.iloc[0, 2])
col4.subheader('Structure ID:')
col4.write(df_protein.iloc[0, 3])
col5.subheader('Mode of action:')
col5.write(df_protein.iloc[0, 4])


# Erstellen des Buttons zur Vorhersage

st.write('<style>div.row-widget.stButton>button{background-color: #F4F3EC; color: #B318AE; font-size: 44px;}</style>', unsafe_allow_html=True)
if st.button("**Are you the one?**", use_container_width=True):
    # Extrahieren der Features des ausgewählten Proteins
    X = df_protein
    
    # Vorhersage der Druggability des Proteins
    prediction = model.predict(X)[0]
    
    # Anzeigen der Vorhersage
    if prediction == 0:
        st.write(f"The selected protein ({protein_choice}) is **not** druggable.")
    else:
        st.write(f"The selected protein ({protein_choice}) is druggable.")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')


import webbrowser

if st.button("Check your prediction here"):
    if protein_choice == 'T81850':
        url = 'https://db.idrblab.net/ttd/search/ttd/target?search_api_fulltext=T81850'
    elif protein_choice == 'T85943':
        url = 'https://db.idrblab.net/ttd/search/ttd/target?search_api_fulltext=T85943'
    webbrowser.open_new_tab(url)

# # Vorhersagefunktion
# def predict(df):
#     # Hier wird der Code eingefügt, der die Vorhersage durchführt
#     # Wir gehen davon aus, dass das Modell eine Methode "predict" hat, die die Vorhersage durchführt
#     # und eine Liste von Vorhersageergebnissen zurückgibt.
#     predictions = model.predict(df)
#     return predictions

# # Bestätigungs-Button
# if st.button("Are you the one?"):
#     # Aufruf der Vorhersagefunktion mit dem entsprechenden DataFrame
#     results = predict(df)

# # Ausgabe des Ergebnisses je nach Protein-ID-Auswahl
#     if protein_choice == "T81850":
#         if results[0] == 0:
#             st.write("Das Protein T81850 ist nicht druggable.")
#         else:
#             st.write("Das Protein T81850 ist druggable.")
#     else:
#         if results[0] == 0:
#             st.write("Das Protein T85943 ist nicht druggable.")
#         else:
#             st.write("Das Protein T85943 ist druggable.")




###########################################
# Add presentation for "Tools":




###########################################
# Add presentation for "Thanks":