from email.header import Header
from turtle import color
import cv2
import numpy as np
from requests import head
import streamlit as st
import base64
import keras
import tensorflow as tf

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocress_input
from PIL import Image

# ทำด้านข้าง
with st.sidebar:
    st.title('Contact:')
    st.header('Phatcharaphon Ubonratsame')
    st.header(" ")
    st.write('''
    
    Tel : 0901470286''', '''Email : kong.04@windowslive.com''',
    '''
    
    Adress : 911, Moo 9,Mueang Nakhon Sawan District, Nakhonsawan, 60000
    

    ''')
    
    st.header(" ")
    st.header(" ")
    Im1 = Image.open('NULOGO-Download-EN.png')
    Im2 = Image.open('logo-sci-01.webp')
    st.image([Im1, Im2] , width= 150)
    
    
page_bg_img = """ 
<style> 
[data-testid="stAppViewContainer"] > .main {                                                                                                                                                                                                         
background-image: url("https://static.vecteezy.com/system/resources/previews/006/852/804/original/abstract-blue-background-simple-design-for-your-website-free-vector.jpg");
background-size: cover; 
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
right: 2rem;
}
</style> 
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
head1 = '<p style="font-family:sans-serif; color:orange; font-size: 42px;">Brain Tumor Classification Web App</p>'
st.markdown(head1, unsafe_allow_html=True)
topic_mean_tunor = '<p style="font-family:sans-serif; color:white;">Brain tumors can be categorized as benign tumors and malignant or malignant tumors Most of the time, the exact cause of the tumor is unknown. Some patients find that they may have risk factors or are caused by heredity. in cases of malignant brain tumors The tumor has spread from other organs. which can cause brain tumors Brain tumors can occur in people of any age. especially those aged 40 and over.Brain tumors are abnormally growing tissues in brain cells. or near the brain until affecting the functioning of the brain and nervous system cause various symptoms in the body</p>'
st.markdown(topic_mean_tunor, unsafe_allow_html=True)
img1 = Image.open('parts of brain diagram.jpg')
st.image([img1] )
case = '<p style="font-family:sans-serif; color:white;">In this study, we focused on three tumor types: glioma, meningioma, pituitary tumor using the CNN model to predict the data.</p>'
st.markdown(case, unsafe_allow_html=True)
end = '<p style="font-family:sans-serif; color:white;">The training and testing accuracy for Brain Tumor dataset are 99.21% </p>'


model = load_model('Tumor_CNN_modeldropout0.2.h5')

uploaded_file = st.file_uploader('choose a image file', type = ['png','jpg','tiff'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes ,1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(64,64))
    x = image.img_to_array(resized)
    x /= 255
    x = np.expand_dims(x, axis = 0)

    st.image(opencv_image, channels='RGB')
    images = np.vstack([x])
    classes = model.predict(images)
    
    Generate_pred = st.button('Generate Prediction')

    if Generate_pred:
        if np.argmax(classes) == 0:
            st.title('%s , %.2f'%('glioma',(classes[0][np.argmax(classes)]*100)))
            st.title(": " + 'glioma')
            st.title(classes)
        elif np.argmax(classes) == 1:
            st.title('%s , %.2f'%('meningioma',(classes[0][np.argmax(classes)]*100)))
            st.title( ": " + 'meningioma')
            st.title(classes)
        else:
            st.title('%s , %.2f'%('pituitary tumor',(classes[0][np.argmax(classes)]*100)))
            st.title(": " + 'pituitary tumor')
            st.title(classes)