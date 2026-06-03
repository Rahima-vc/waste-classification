import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model("waste_classifier_cnn.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
img_path = "test.jpg" 
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0   
img_array = np.expand_dims(img_array, axis=0)  

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Prediction Probabilities:", prediction)
print("Predicted Class:", predicted_class)
