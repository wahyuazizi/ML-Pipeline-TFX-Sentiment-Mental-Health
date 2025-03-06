
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'status'
FEATURE_KEY = 'statement'
def transformed_name(key):
    return key + "_xf"

def preprocessing_fn(inputs):
    
    outputs = {}
    # Pastikan teks dalam bentuk DenseTensor sebelum transformasi
    text_input = inputs[FEATURE_KEY]
    # if isinstance(text_input, tf.SparseTensor):
    #     text_input = tf.sparse.to_dense(text_input, default_value="")

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(text_input)
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
