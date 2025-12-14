import tensorflow as tf
import config

# Define the custom object for the fix we made earlier
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

try:
    # Load Keras model with custom object
    model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D})
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save
    with open('mobilenet/model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Success: Model converted to mobilenet/model.tflite")

except Exception as e:
    print(f"Error: {e}")
