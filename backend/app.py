from flask import Flask, request, jsonify
from flask_cors import CORS
from random import choice
from PIL import Image

import numpy as np
import tensorflow as tf
import io
import base64


# Load model
model = tf.keras.models.load_model("../models/doodle_classifier.h5")
class_names = ['cat', 'house', 'tree']

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        selected_style = data.get('style', 'sketch')  # Default to sketch

        print(f"🎨 Received style: {selected_style}")

        # Process image
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

        # Predict
        prediction = model.predict(image_array)[0]
        predicted_label = class_names[np.argmax(prediction)]

        return jsonify({
            "prediction": predicted_label,
            "style": selected_style
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/story', methods=['POST'])
def story():
    try:
        label = request.json['label']
        
        # Simple hardcoded story templates (replace later with GPT or HuggingFace)
        templates = {
            "cat": [
                "Once upon a time, a clever cat roamed the streets of Paris with a tiny red beret.",
                "In a quiet village, a cat became the hero of a bakery when it saved the flour from mice."
            ],
            "house": [
                "There stood a lonely house on a hill, hiding a secret that changed the town forever.",
                "The house had no doors, only stories carved into every brick."
            ],
            "tree": [
                "Deep in the forest, an ancient tree whispered secrets to those who dared to listen.",
                "A tree that glowed at night became the heart of a hidden fairy kingdom."
            ]
        }

        story_text = choice(templates.get(label, ["This is a story about something amazing."]))
        return jsonify({"story": story_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
