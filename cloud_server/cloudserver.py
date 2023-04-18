from sklearn.linear_model import LogisticRegression
import codecs
from flask import Flask, request
import time
import requests
import json

app = Flask(__name__)

# Rest API POST method to update global weights as weighted average
@app.route("/updateGlobals", methods=['POST'])
def do_gobalLocal():
    print("Global Update Started")
    json_data = request.get_json()
    print(json_data['fog_node'])


    fog_path = ""
    start_time = time.time()
    fog_node = request.get_json()
    # the directory that contain the train images set
    file_path = '/app/SharedStorage/model/global_model/model.json'
    fog_1_path = '/app/SharedStorage/model/fog_model_1/model.json'
    fog_2_path = '/app/SharedStorage/model/fog_model_2/model.json'

    # Checking where the update came from
    if json_data['fog_node'] == "fog_node_1":
        fog_path = fog_1_path
    elif json_data['fog_node'] == "fog_node_2":
        fog_path = fog_2_path
    else:
        fog_path = fog_1_path

    # Load global model
    jstring = codecs.open(file_path, 'r', encoding='utf-8').read()
    server_model_data = json.loads(jstring)
    log_reg_global = LogisticRegression(**server_model_data['init_params'])

    # Load the corresponding fog node model
    jstring_fog = codecs.open(fog_path, 'r', encoding='utf-8').read()
    fog_model_data = json.loads(jstring_fog)

    avg_w = []

    # Calculate the average of the weights
    for m1, m2 in zip(server_model_data['coef'][0], fog_model_data['coef'][0]):
        avg = (m1 + m2) / 2
        avg_w.append(avg)

    # Calculate the average of the bias
    avg_bias = (server_model_data['intercept'][0] + fog_model_data['intercept'][0]) / 2

    # Update the global model with the newly evaluated weights and bias
    avg_model_param = {}
    avg_model_param['init_params'] = log_reg_global.get_params()
    avg_model_param['coef'] = [avg_w]
    avg_model_param['intercept'] = [avg_bias]
    avg_model_param['classes'] = server_model_data['classes']
    json.dump(avg_model_param, codecs.open(file_path, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)
    print("Global Update Completed")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Trigger fog nodes, to inform them to get the latest globally updated models
    response_f_1 = requests.post(url="http://0.0.0.0:9191/getGlobalUpdate")
    response_f_2 = requests.post(url="http://0.0.0.0:9292/getGlobalUpdate")
    print("Triggered FOG nodes to pu Global Updates")
    return "Global Update Completed"

code = {'NORMAL':0 ,'PNEUMONIA':1}
#function to return the class of the images from its number, so the function would return 'Normal' if given 0, and 'PNEUMONIA' if given 1.
def getcode(n) :
    for x , y in code.items() :
        if n == y :
            return x

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)