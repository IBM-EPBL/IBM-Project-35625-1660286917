from flask import  Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open("histmodel.pkl","rb"))
@app.route("/")
def home():
    return render_template("newindex.html")

@app.route('/submit',methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        yearofRegistration=request.form["yearofRegistration"]
        powerPS=request.form["powerPS"]
        kilometer=request.form["kilometer"]
        monthofRegistration=request.form["monthofRegistration"]
        namelen=request.form["namelen"]
        gearbox_feat=request.form["gearbox_feat"]
        if gearbox_feat == "manuell":
            gearbox_feat =1
        elif gearbox_feat == "auto":
            gearbox_feat =0
        fuelType_feat = request.form["fuelType_feat"]
        if fuelType_feat=="petrol":
            fuelType_feat=1
        elif fuelType_feat=="benzin":
            fuelType_feat=2
        elif fuelType_feat=="diesel":
            fuelType_feat=3
        elif fuelType_feat=="lgp":
            fuelType_feat=4
        elif fuelType_feat=="andere":
            fuelType_feat=5
        elif fuelType_feat=="hybrid":
            fuelType_feat=6
        elif fuelType_feat=="cng":
            fuelType_feat=7
        elif fuelType_feat=="elektro":
            fuelType_feat=8
        int_features = [yearofRegistration, powerPS,kilometer,monthofRegistration,namelen,gearbox_feat,fuelType_feat]
        features = [np.array(int_features, dtype=int)]
        prediction=model.predict(features)
        return render_template("newsubmit.html",prediction=round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
