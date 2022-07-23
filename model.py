from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras import initializers, regularizers, constraints, optimizers, layers
import streamlit as st

loaded_model_1 = keras.models.load_model(filepath='C:/Users/aarth/Downloads/jigsaw-toxic-comment-classification-challenge/toxicity1.h5')


max_features=100000      
maxpadlen = 200          
val_split = 0.2      
embedding_dim_fasttext = 300




def toxicity_level(string):
    tokenizer= Tokenizer()
    new_string = [string]
    new_string = tokenizer.texts_to_sequences(new_string)
    new_string = pad_sequences(new_string, maxlen=maxpadlen, padding='post')
    
    prediction = loaded_model_1.predict(new_string) #(Change to model_1 or model_2 depending on the preference of model type|| Model 1: LSTM, Model 2:LSTM-CNN)
    
    st.write("Toxicity levels for '{}':".format(string))
    st.write('Toxic:         {:.0%}'.format(prediction[0][0]))
    st.write('Severe Toxic:  {:.0%}'.format(prediction[0][1]))
    st.write('Obscene:       {:.0%}'.format(prediction[0][2]))
    st.write('Threat:        {:.0%}'.format(prediction[0][3]))
    st.write('Insult:        {:.0%}'.format(prediction[0][4]))
    st.write('Identity Hate: {:.0%}'.format(prediction[0][5]))
  