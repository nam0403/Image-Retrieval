import h5py
from keras.applications.vgg19 import preprocess_input
import keras.utils as image
import numpy as np
import os
import cv2 as cv
from keras.models import Model
from tensorflow.keras.applications import Xception
from scipy import spatial
from PIL import Image
import argparse
from keras.applications.vgg19 import VGG19

base_model = VGG19(weights='imagenet',include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

folder_path = 'dataset'
image_ext = ('jpg', 'jpeg', 'png')
img_paths = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.endswith(image_ext)]

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True, help="query to search")
args = parser.parse_args()

with h5py.File('features_vgg19.h5', 'r') as data:
    features = data.get('features')
    dataset = features[...]
        
# def add_index(dataset):
#     index = AnnoyIndex(131072, metric='angular')
#     for i, img_features in enumerate(dataset):
#         index.add_item(i, img_features)
#     return index
# def calculate_score(query_feature, database_feature):
#     score = euclidean(query_feature, database_feature)
#     return score
def load_image_query(img):
    img = cv.resize(img,(256,256))
    x = image.img_to_array(img)
    # reshape data for the model
    x = np.expand_dims(x, axis=0)
    # preprocess inputs
    x = preprocess_input(x)
    # get features
    # features = model.predict(x)
    feature = model.predict(x)[0]
    feature = feature.flatten()
    feature = feature / np.linalg.norm(feature)
    return feature
# def search_image(feature_vector, n , search_k, include_distances):
#     index = add_index(dataset)
#     index.build(20)
#     img_ids, distances = index.get_nns_by_vector(feature_vector, n, search_k, include_distances)
#     results = [(ID, distance) for ID, distance in zip(img_ids, distances)]
#     return results
    
queryImage = cv.imread(args.query)
query_feature = load_image_query(queryImage)
# results = search_image(feature_vector, 10, 100, True)

# Function for finding nearest neighbor
def find_nearest_neighbor(query_feature, features, top_k=1):
    similarity = []
    for vector in features:
        cos_sim = 1 - spatial.distance.cosine(vector, query_feature)
        similarity.append(cos_sim)
        # select top K relevant image
        ids = np.argsort(similarity)[::-1][:top_k]
    relevant_imgs = [(similarity[id], img_paths[id]) for id in ids]
    return relevant_imgs

# Find the 10 nearest neighbors to a given query feature
results = find_nearest_neighbor(query_feature, dataset, top_k=10)
# for similarity, img_path in results:
#     img = Image.open(img_path)
#     img.show()
    
import csv
fields = ['similarity', 'img_path']
# data rows of csv file
rows = [[similarity, img_path] for similarity, img_path in results ]
# name of csv file
filename = "results.csv"
# writing to csv file
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)
data.close()