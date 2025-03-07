
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'label'
FEATURE_KEY = 'text'
def transformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
    
    outputs = {}
    # Pastikan teks dalam bentuk DenseTensor sebelum transformasi
    text_input = inputs[FEATURE_KEY]
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(text_input)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
