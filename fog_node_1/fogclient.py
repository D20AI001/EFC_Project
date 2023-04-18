import numpy as np
import cv2

import requests
from sklearn.linear_model import LogisticRegression
import json
import codecs
from sklearn.utils import shuffle as shf
import os
import glob as gb
from flask import Flask
import shutil
import time

from threading import Thread

app = Flask(__name__)


# Rest API POST method to update local model
@app.route("/updateLocal", methods=['POST'])
def do_updateLocal():
    print("FOG model update started")
    start_time = time.time()

    # the directory that contain the train images set
    file_path = '/app/SharedStorage/model/fog_model_1/model.json'
    edge_path = '/app/SharedStorage/model/edge_model_1/model.json'

    # Load the fog nodes model
    jstring = codecs.open(file_path, 'r', encoding='utf-8').read()
    data = json.loads(jstring)
    fog_model = LogisticRegression(**data['init_params'])
    setattr(fog_model, 'coef_', np.array(data['coef']))
    setattr(fog_model, 'intercept_', np.array(data['intercept']))
    setattr(fog_model, 'classes_', np.array(data['classes']))
    # setattr(fog_model, 'n_iter_', np.array(data['iter']))

    # Path to train folder
    trainpath = '/app/SharedStorage/data/chest_xray/train/'

    X_train = []
    y_train = []
    # Create numpy arrays of train data and train labels
    for folder in os.listdir(trainpath):
        files = gb.glob(pathname=str(trainpath + folder + '/*.jpeg'))
        for file in files:
            image = cv2.imread(file)
            # resize images to 64 x 64 pixels
            image_array = cv2.resize(image, (64, 64))
            X_train.append(list(image_array))
            y_train.append(code[folder])
    np.save('X_train', X_train)
    np.save('y_train', y_train)

    # the directory that contain the test images set
    testpath = '/app/SharedStorage/data/chest_xray/test/'

    X_test = []
    y_test = []
    # Create numpy arrays of test data and test labels
    for folder in os.listdir(testpath):
        files = gb.glob(pathname=str(testpath + folder + '/*.jpeg'))
        for file in files:
            image = cv2.imread(file)
            # resize images to 64 x 64 pixels
            image_array = cv2.resize(image, (64, 64))
            X_test.append(list(image_array))
            y_test.append(code[folder])
    np.save('X_test', X_test)
    np.save('y_test', y_test)

    # X_train, X_test contain the images as numpy arrays, while y_train, y_test contain the class of each image
    loaded_X_train = np.load('./X_train.npy')
    loaded_X_test = np.load('./X_test.npy')
    y_train = np.load('./y_train.npy')
    y_test = np.load('./y_test.npy')

    # print(loaded_X_train.shape)
    # print(y_train.shape)
    # print(loaded_X_test.shape)
    # print(y_test.shape)

    # flatten the images into a 2d array, for model training and testing
    X_train = loaded_X_train.reshape([-1, np.product((64, 64, 3))])
    X_test = loaded_X_test.reshape([-1, np.product((64, 64, 3))])

    # shuffle train and test data sets in a consistent way
    X_train, y_train = shf(X_train, y_train, random_state=15)
    X_test, y_test = shf(X_test, y_test, random_state=15)

    # Train the model with image obtained from the edge device
    fog_model.fit(X_train, y_train)

    accuracy = str(fog_model.score(X_test, y_test))

    model_param = {}
    model_param['init_params'] = fog_model.get_params()
    model_param['coef'] = fog_model.coef_.tolist()
    model_param['intercept'] = fog_model.intercept_.tolist()
    model_param['classes'] = fog_model.classes_.tolist()
    model_param['iter'] = fog_model.n_iter_.tolist()

    # Save the newly learned weights in the shared storage
    json.dump(model_param, codecs.open(file_path, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)

    # Place the new model weights in the storage location that edge device will refer to for inferences
    shutil.copy(file_path, edge_path)
    print(accuracy)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("FOG model update completed")
    Thread(target=sendTriggerToServer).start()

    return accuracy


# Rest API with POST method, this call is made by global cloud server to let the local fog node know that a new model
# is available for pull
@app.route("/getGlobalUpdate", methods=['POST'])
def do_getGlobal():
    print("Trigger to get latest global updates")
    server_path = '/app/SharedStorage/model/global_model/model.json'
    fog_node_path = '/app/SharedStorage/model/fog_model_1/model.json'
    edge_path = '/app/SharedStorage/model/edge_model_1/model.json'
    shutil.copy(server_path, fog_node_path)
    shutil.copy(server_path, edge_path)
    print("Local Model updated with Global weights")
    return "Update Complete"


code = {'NORMAL': 0, 'PNEUMONIA': 1}


# function to return the class of the images from its number, so the function would return 'Normal' if given 0,
# and 'PNEUMONIA' if given 1.
def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

# Method to send a trigger to cloud server to let it know that a new model is available for aggregation
def sendTriggerToServer():
    fog_data = {"fog_node": "fog_node_1"}
    print("Trigger cloud server for global update")
    response_f_1 = requests.post(url="http://0.0.0.0:9090/updateGlobals", json=fog_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9191)
