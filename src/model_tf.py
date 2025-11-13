# src/model_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_convlstm_relu(input_shape, out_frames=1, filters=(32, 64),
                        kernels=(3, 3), use_bn=True, dropout=0.1, relu_cap=1.0):
    """
    input_shape: (T_in, H, W, 1)
    out_frames: số khung dự đoán (thường = 1)
    """
    x_in = layers.Input(shape=input_shape)  # (T_in,H,W,1)
    x = x_in
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(
            filters=f,
            kernel_size=(kernels[i], kernels[i]),
            padding="same",
            return_sequences=(i < len(filters) - 1),
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(out_frames, 3, padding="same")(x)
    x = layers.ReLU(max_value=relu_cap)(x)                  # ép [0,1] nếu relu_cap=1
    x = layers.Lambda(lambda t: tf.expand_dims(t, 1))(x)    # (B,1,H,W,1)
    return models.Model(x_in, x)

def r2_metric(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    y_mean = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - y_mean))
    return 1.0 - tf.math.divide_no_nan(ss_res, ss_tot)

def compile_model(model, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        r2_metric
    ]
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=metrics
    )
    return model
