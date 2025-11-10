import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def r2_metric(y_true, y_pred):
    # R² = 1 - SS_res / SS_tot (thêm epsilon để tránh chia 0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_true = tf.reduce_mean(y_true)
    ss_tot = tf.reduce_sum(tf.square(y_true - mean_true))
    return 1.0 - ss_res / (ss_tot + 1e-8)

def build_convlstm_relu(input_shape,
                        out_frames=1,           # số frame thời gian output (báo cáo đang dùng 1)
                        filters=(32, 64),
                        kernels=(3, 3),
                        use_bn=True,
                        dropout=0.1,
                        relu_cap=1.0):
    """
    input_shape: (T_in, H, W, 1)  — ví dụ (4, 128, 128, 1) hoặc (64, 128, 128, 1)
    out_frames: số frame thời gian dự báo. Hiện kiến trúc này trả 1 frame theo chiều thời gian,
                nên để out_frames=1 (nếu muốn >1 thì phải đổi sang head 3D khác).
    """
    x_in = layers.Input(shape=input_shape)  # (T_in, H, W, 1)
    x = x_in

    # ConvLSTM blocks
    for i, f in enumerate(filters):
        x = layers.ConvLSTM2D(
            filters=f,
            kernel_size=(kernels[i], kernels[i]),
            padding="same",
            return_sequences=(i < len(filters) - 1),  # True cho tất cả trừ block cuối
            activation="tanh",
            kernel_regularizer=regularizers.l2(1e-6),
            recurrent_dropout=0.0
        )(x)

        if use_bn:
            x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    # Projection ra bản đồ 1 kênh, sau đó kẹp 0..relu_cap (thường =1.0 cho dữ liệu đã chuẩn hóa [0,1])
    # Lưu ý: kiến trúc hiện tại xuất 1 frame thời gian. Nếu muốn out_frames>1 theo thời gian,
    # cần thay head bằng Conv3D hoặc dùng ConvLSTM return_sequences + TimeDistributed(Conv2D).
    x = layers.Conv2D(1, kernel_size=3, padding="same")(x)     # 1 kênh cường độ
    x = layers.ReLU(max_value=relu_cap)(x)                      # kẹp về [0,1] nếu relu_cap=1.0

    # Thêm trục thời gian = 1: (B, H, W, 1) -> (B, 1, H, W, 1)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(x)

    model = models.Model(x_in, x, name="ConvLSTM_ReLU_cap")
    return model

def compile_model(model, lr=1e-3, loss="mse"):
    """
    Báo cáo đang lấy val_loss = MSE, nên mặc định loss='mse'.
    Nếu Yoshino muốn Huber thì đặt loss=tf.keras.losses.Huber(delta=0.05),
    nhưng lúc đó val_loss không còn là MSE nữa.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.RootMeanSquaredError(),
        r2_metric
    ],
)


    return model
