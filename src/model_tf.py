import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_convlstm_relu(input_shape, out_frames=1, filters=(32,64),
                        kernels=(3,3), use_bn=True, dropout=0.1, relu_cap=1.0):
    """
    input_shape: (T_in, H, W, 1)
    out_frames: 1 khung dự đoán
    """
    x_in = layers.Input(shape=input_shape)  # (T_in,H,W,1)
    x = x_in
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(
            filters=f,
            kernel_size=(kernels[i], kernels[i]),
            padding="same",
            return_sequences=(i < len(filters)-1),
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-6)
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    # lớp ra: ReLU theo yêu cầu bài
    x = layers.Conv2D(out_frames, 3, padding="same")(x)
    x = layers.ReLU(max_value=relu_cap)(x)   # khóa 0..1 nếu relu_cap=1.0
    x = layers.Lambda(lambda t: tf.expand_dims(t, 1))(x)  # (B,1,H,W,1)
    return models.Model(x_in, x)

def compile_model(model, lr=1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    # Huber hợp với ReLU và dữ liệu minmax
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=0.05))
    return model
