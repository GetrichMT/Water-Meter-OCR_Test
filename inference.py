from tensorflow import keras
import numpy as np
from preprocess import preprocess_image, num_to_char

def decode_output(preds):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = keras.backend.ctc_decode(preds, input_length=input_len)
    result = []
    for seq in decoded[0].numpy():
        text = ''.join([num_to_char.get(i, '') for i in seq if i != -1])
        result.append(text)
    return result

# Load model
model = keras.models.load_model("model.h5", compile=False)

# Load test image
img = preprocess_image("images/021232.jpg")
img = np.expand_dims(img, axis=0)

# Predict and decode
preds = model.predict(img)
print("Predicted:", decode_output(preds)[0])
