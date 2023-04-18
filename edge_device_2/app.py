
from __future__ import division, print_function
import random

import os
import requests

from tensorflow.keras.preprocessing import image

import cv2
import json
import codecs
from sklearn.linear_model import LogisticRegression
import numpy as np
from threading import Thread
# Flask utils
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
# Define a flask app
app = Flask(__name__)

# Method for prediction of the lung condition
def model_predict(img_path):


    #pic = Image.open(img_path)
    # Load the image, resized to 150 x 150
    img = image.load_img(img_path, target_size=(150, 150))
    array = image.img_to_array(img)
    x = np.expand_dims(array, axis=0)
    file_path = '/home/sagar/Main/SharedStorage/model/edge_model_2/model.json'

    # load the pre trained model from the local directory
    jstring = codecs.open(file_path, 'r', encoding='utf-8').read()
    data = json.loads(jstring)
    log_reg_new = LogisticRegression(**data['init_params'])
    setattr(log_reg_new, 'coef_', np.array(data['coef']))
    setattr(log_reg_new, 'intercept_', np.array(data['intercept']))
    setattr(log_reg_new, 'classes_', np.array(data['classes']))
    #setattr(log_reg_new, 'n_iter_', np.array(data['iter']))

    # Load the image, resized to 64 x 64
    image_test = cv2.imread(img_path)
    image_array_test = cv2.resize(image_test, (64, 64))
    np.save('X_pred', list(image_array_test))
    loaded_X_pred = np.load('./X_pred.npy')
    X_pred = loaded_X_pred.reshape([-1, np.product((64, 64, 3))])
    # Pass the numpy array to the model for prediction
    log_reg_pred = log_reg_new.predict(X_pred)
    result = getcode(log_reg_pred)

    # A seperate thread to trigger fog node for model update in an asynchonous way
    Thread(target=upl_new_data_to_fog,
                     args=(img_path,result)).start()

    return "Predicted Lung condition is : "+ result


code = {'NORMAL': 0, 'PNEUMONIA': 1}


# REST api with GET method to render the index page
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# REST api with POST method to to predict the lung condition
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        result=preds
        return result
    return None

#function to return the class of the images from its number, so the function would return 'Normal' if given 0, and 'PNEUMONIA' if given 1.
def getcode(n) :
    for x , y in code.items() :
        if n == y :
            return x

# Meethod to make an API call oto fog node to update its local model
def upl_new_data_to_fog(image,label):
    pnuemonia_path = '/home/sagar/Main/SharedStorage/data/chest_xray/train/PNEUMONIA'
    normal_path = '/home/sagar/Main/SharedStorage/data/chest_xray/train/NORMAL'
    image_path_tain = ""
    print(image)
    if label == 'PNEUMONIA':
        image_path_tain = pnuemonia_path
    else:
        image_path_tain = normal_path
    imge_name = os.path.basename(image).split('/')[-1]
    pic = Image.open(image)
    # Place the image in the train folder of the fog node

    im1 = pic.save(image_path_tain+"/"+imge_name)
    response = requests.post(url="http://0.0.0.0:9292/updateLocal")
    os.remove(image)
    return "done"

if __name__ == '__main__':
    app.run(debug=True)
