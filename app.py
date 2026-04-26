from flask import Flask, request, render_template
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

X = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
])

y = np.array([1, 1, 1, 0, 0])

model = LogisticRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fever = int(request.form['fever'])
    cough = int(request.form['cough'])
    fatigue = int(request.form['fatigue'])

    prediction = model.predict([[fever, cough, fatigue]])

    result = "Flu" if prediction[0] == 1 else "Cold"
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)