from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('student_mark_predictor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        hours = float(request.form['hours'])
        prediction = model.predict(np.array([[hours]]))[0]
        if prediction > 100:
            prediction = 100  # Cap the grade at 100
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)