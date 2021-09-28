import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model1=pickle.load(open("model1.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    list_features = [int(x) for x in request.form.values()]
    np_features = np.array(list_features).reshape(1,-1)
    prediction1 = model1.predict(np_features)
    if(prediction1[0]==0):
        opt="Balanced"
    if(prediction1[0]==1):
        opt="Left"
    if(prediction1[0]==2):
        opt="Right"
    return render_template('index.html', otptext="The scale is {}".format(opt))

if __name__ == "__main__":
    app.run(debug=True)