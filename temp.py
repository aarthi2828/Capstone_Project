import streamlit as st
from model import toxicity_level

input = st.text_input('Enter comment')
ok = st.button('Submit')
if ok:
    toxicity_level(input)
    
    
    
    

