import h5py
from keras.applications.vgg16 import VGG16
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from annoy import AnnoyIndex
import os
from streamlit_cropper import st_cropper
from PIL import Image
import streamlit as st
from sklearn.neighbors import KDTree
import argparse
import time

start_time = time.time()
if not os.path.exists('dataset'):
    os.makedirs('dataset')
if st.button('Get dataset'):
    parser = argparse.ArgumentParser(description='Process some dataset.')
    parser.add_argument('--dataset_folder', type=str, default='dataset', help='Path to dataset folder')
    args = parser.parse_args()
    os.system(f'python get_dataset.py --dataset_folder {args.dataset_folder}')
    end_time = time.time()
    st.success("Dataset download complete!")
    st.write(f'Time taken to download dataset: {end_time - start_time} seconds')
        
# img = cv.imread('query.jpeg')
# image_query = load_image_query(img)
# search_image(image_query, 5, 100, True)

uploaded_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    cropped_img = st_cropper(input_image, realtime_update=realtime_update, box_color=box_color)
    if st.button('Crop and Query'):
        cropped_img.save("test.png")
        query_path = 'test.png'
        os.system('python improve.py --query {}'.format(query_path))
        import pandas as pd 
        results = pd.read_csv('./results.csv')
        #st.dataframe(results)
        for similarity, img_path in zip(results['similarity'], results['img_path']):
            img = Image.open(img_path)
            st.image(img, channels="BGR")
            # st.write(score)
