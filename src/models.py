from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW

def conv_block(x, filters, k, pool=True):
    sc = x
    x = layers.Conv1D(filters, k, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, k, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if sc.shape[-1] != filters:
        sc = layers.Conv1D(filters, 1, padding='same', use_bias=False)(sc)
        sc = layers.BatchNormalization()(sc)
    x = layers.Add()([x, sc])
    x = layers.ReLU()(x)
    if pool:
        x = layers.MaxPool1D(2)(x)
    return x

def build_iq_cnn_res(n_classes, input_len):
    inp = layers.Input(shape=(input_len, 2))
    x = conv_block(inp, 128, 7, pool=True)
    x = conv_block(x,   192, 5, pool=True)
    x = conv_block(x,   256, 3, pool=True)
    x = conv_block(x,   256, 3, pool=False)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)

model = build_iq_cnn_res(n_classes=len(MODS), input_len=INPUT_LEN)
model.summary()
