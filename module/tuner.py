
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from indo_tele_sentiment import model_builder, input_fn

lABEL_KEY = "label"
FEATURE_KEY = "text"


def tuner_fn(fn_args):
    # Membuat training dan validation datasetyang telah di preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=10)

    # Callback untuk early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.Hyperband(
        model_builder,
        objective = 'val_accuracy',
        max_epochs = 10,
        factor = 3,
        directory = fn_args.working_dir,
        project_name = 'kt_hyperband'
    )

    return TunerFnResult(
        tuner = tuner,
        fit_kwargs = {
            "callbacks": [stop_early],
            'x' : train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
