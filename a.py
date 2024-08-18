#=======================================================================
## 0. Importing libraries and setting up streamlit web app

#Importing the necessary packages
import streamlit as st
import openpyxl
import pygwalker as pyg
import pandas as pd

#Setting up web app page
st.set_page_config(page_title='Exploratory Data Analysis App', page_icon=None, layout="wide")