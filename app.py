import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image ,ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

@st.cache(allow_output_mutation=True)

def predict(img):
    IMAGE_SIZE = 224
    classes = ['Apple - Apple scab', 'Apple - Black rot',
    'Apple - Cedar apple rust', 'Apple - healthy', 'Background without leaves',
    'Blueberry - healthy', 'Cherry - Powdery mildew', 'Cherry - healthy',
    'Corn - Cercospora leaf spot Gray leaf spot', 'Corn - Common rust',
    'Corn - Northern Leaf Blight', 'Corn - healthy', 'Grape - Black rot',
    'Grape - Esca (Black Measles)', 'Grape - Leaf blight (Isariopsis Leaf Spot)',
    'Grape - healthy', 'Orange - Haunglongbing (Citrus greening)',
    'Peach - Bacterial spot', 'Peach - healthy', 'Pepper, bell - Bacterial spot',
    'Pepper, bell - healthy', 'Potato - Early blight', 'Potato - Late blight',
    'Potato - healthy', 'Raspberry - healthy', 'Soybean - healthy',
    'Squash - Powdery mildew', 'Strawberry - Leaf scorch', 'Strawberry - healthy',
    'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot',
    'Tomato - Spider mites Two-spotted spider mite', 'Tomato - Target Spot',
    'Tomato - Tomato Yellow Leaf Curl Virus', 'Tomato - Tomato mosaic virus',
    'Tomato - healthy']
    model_path = r'model'
    model = tf.keras.models.load_model(model_path)
    img = Image.open(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img)
    img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    img = img/255.
    class_probabilities = model.predict(x=img)
    class_probabilities = np.squeeze(class_probabilities)
    prediction_index = int(np.argmax(class_probabilities))
    prediction_class = classes[prediction_index]
    prediction_probability = class_probabilities[prediction_index] * 100
    prediction_probability = round(prediction_probability, 2)
    return prediction_class, prediction_probability


def load_model():
    model=tf.keras.models.load_model('my_model.hdf5')
    return model

model2=load_model()

def import_and_predict(image_data , model):
    size=(256,256)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model2.predict(img_reshape)
    return prediction


st.markdown('<style>body{text-align: center;}</style>', unsafe_allow_html=True)

# Main app interface
st.title('plant and soil Classification ')
st.image('appimage.jpg',width=600, height=500)

img = st.file_uploader(label='Upload leaf image (PNG, JPG or JPEG)', type=['png', 'jpg', 'jpeg'])

st.write('Please specify the type of classifier (soil or plant)')
if img is not None:
    predict_button = st.button(label='Plante Disease Classifier')
    prediction_class, prediction_probability = predict(img)
    if predict_button:
        st.image(image=img.read(), caption='Uploaded image')
        
        st.subheader('Prediction')
        st.info(f'Classification: {prediction_class}, Accuracy: {prediction_probability}%')
    
    predict_button2 = st.button(label='Soil Classifier')
    if predict_button2:
        image=Image.open(img)
        st.image(image,size=(350,350))
        st.subheader('Prediction')
        predictions=import_and_predict(image,model2)
        class_names=['clay soil', 'gravel soil', 'loam soil', 'sand soil']
        score = tf.nn.softmax(predictions[0])
        st.info(f'Classification: {class_names[np.argmax(predictions)]}, Accuracy: { 100 * np.max(score)}%')
       

        
