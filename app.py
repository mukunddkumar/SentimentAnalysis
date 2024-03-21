from flask import Flask, render_template, request
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        model_path = os.path.join(os.path.dirname(__file__), "Models", "naive_bayes.pkl")
        rf_model = joblib.load(model_path)
        prediction = rf_model.predict([review])[0]
        if prediction == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
        return render_template('output.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
