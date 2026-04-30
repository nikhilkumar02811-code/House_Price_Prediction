from flask import Flask,render_template,request 
import joblib 
import numpy as np
from sklearn.linear_model import LinearRegression

model = joblib.load("model.pkl")



app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("myform.html")

@app.route("/predict",methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedroom = float(request.form["bedroom"])
    age = float(request.form["age"])


    feature = np.array([[area,bedroom,age]])
    prediction= model.predict(feature)

    return render_template("myform.html",prediction_text=f"house price is{prediction[0]}")


if __name__ == "__main__":
    app.run(debug=True)
