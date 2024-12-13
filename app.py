from flask import Flask, request, render_template, url_for
import numpy as np
import pickle

# Importing model
model = pickle.load(open('information/model.pkl', 'rb'))
sc = pickle.load(open('information/standscaler.pkl', 'rb'))
ms = pickle.load(open('information/minmaxscaler.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__, template_folder='Templates')  # Set the correct template folder

@app.route('/')
def index():
    return render_template("index.html")  # Reference index.html in Templates

@app.route('/info')
def info():
    return render_template("info.html")  # Reference info.html in Templates

@app.route("/predict", methods=['POST'])
def predict():
    # Extract form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Prepare features for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale and predict
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    # Map prediction to crop and image path
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        image_path = url_for('static', filename=f'{prediction[0]}.jpg')  
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        image_path = None
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result, image_path=image_path)

# Main
if __name__ == "__main__":
    app.run(debug=True)
