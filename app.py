from flask import Flask, request
from PIL import Image
import numpy as np

import ml_service
from ml_model import load_model

app = Flask(__name__)
model = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files.get('image')

    if not image_file:
        return {"error": "Send an image to run prediction"}, 400

    image = Image.open(image_file).convert('L')
    image_array = np.asarray(image)

    prediction = ml_service.predict(image_array.reshape((1,) + image_array.shape),
                                    model)

    return {'modelOutput': prediction,
            'hasTuberculosis': prediction > 0.5}


if __name__ == '__main__':
    app.run(debug=True)
