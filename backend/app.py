from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Load model
model = tf.keras.models.load_model("../models/doodle_classifier.h5")
class_names = ['cat', 'house', 'tree']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(image_array)[0]
        predicted_label = class_names[np.argmax(prediction)]
        return jsonify({"prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
