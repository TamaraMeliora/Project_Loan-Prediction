from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    image_file = None  # для картинки

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            income = float(request.form["income"])
            loan_amount = float(request.form["loanamount"])  # исправлено

            input_data = np.array([[age, income, loan_amount]])
            input_scaled = scaler.transform(input_data)

            pred = model.predict(input_scaled)[0]
            pred_proba = model.predict_proba(input_scaled)[0][1]

            if pred == 1:
                prediction = "⚠️ High risk of default"
                image_file = "default.png"  # картинка для дефолта
            else:
                prediction = "✅ Loan is likely to be repaid"
                image_file = "nofault.png"  # картинка для нормального случая

            proba = f"Default probability: {pred_proba:.2%}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, proba=proba, image_file=image_file)

if __name__ == "__main__":
    app.run(debug=True)
