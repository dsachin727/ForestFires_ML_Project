import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

##import ridge and scaler models
ridge_model = pickle.load(open('Model/ridge.pkl','rb'))
scaler_model = pickle.load(open('Model/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        data = [Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]

        scaled_model = scaler_model.transform([data])
        prediction = ridge_model.predict(scaled_model)
        
        return render_template('home.html', results = prediction)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
#above by default runs on port 5000 but since that port was used on different project/program we manually given different port to run

