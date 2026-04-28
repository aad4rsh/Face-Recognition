import cv2
import os
import numpy as np
from sklearn.datasets import fetch_lfw_people

def get_lfw_data():
    # Fetching LFW dataset (requires internet first time)
    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data # Flattened pixels
    y = lfw_people.target
    target_names = lfw_people.target_names
    
    # Return images, labels, names, and dimensions
    return X, y, target_names, h, w