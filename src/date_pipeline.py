AUTOTUNE = tf.data.AUTOTUNE

# ---- RF augmentations (mild) ----
def aug_phase_shift(x_2xT):
    theta = tf.random.uniform([], -np.pi/16, np.pi/16)
    rot = tf.complex(tf.cos(theta), tf.sin(theta))
    c = tf.complex(x_2xT[0], x_2xT[1]) * rot
    return tf.stack([tf.math.real(c), tf.math.imag(c)], axis=0)

def aug_time_shift(x_2xT):
    shift = tf.random.uniform([], -16, 17, dtype=tf.int32)
    return tf.roll(x_2xT, shift=shift, axis=1)

def aug_noise(x_2xT):
    std = tf.random.uniform([], 0.003, 0.012)
    return x_2xT + tf.random.normal(tf.shape(x_2xT), stddev=std)

# ---- Per-channel standardization; transpose to [T,2] for Conv1D ----
def normalize_iq_2xT(x_2xT):
    mean = tf.reduce_mean(x_2xT, axis=1, keepdims=True)
    std  = tf.math.reduce_std(x_2xT, axis=1, keepdims=True) + 1e-8
    x = (x_2xT - mean) / std
    return tf.transpose(x, [1, 0])  # [T,2]

# ---- Map function: apply aug with probability p on training only ----
def map_example(x_2xT, y, augment=False, p=0.5):
    if augment:
        r = tf.random.uniform([])
        def do_aug():
            z = aug_phase_shift(x_2xT)
            z = aug_time_shift(z)
            z = aug_noise(z)
            return z
        x_2xT = tf.cond(r < p, do_aug, lambda: x_2xT)
    x = normalize_iq_2xT(x_2xT)      # -> [T,2]
    return x, y

def make_dataset(X, Y, batch=512, shuffle=True, augment=False, p=0.5):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(lambda x,y: map_example(x, y, augment=augment, p=p),
                num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# build datasets (we’ll warm-up without aug, then switch to aug)
train_ds_noaug = make_dataset(trainX, trainY, batch=512, shuffle=True,  augment=False)
val_ds         = make_dataset(valX,   valY,   batch=512, shuffle=False, augment=False)
test_ds        = make_dataset(testX,  testY,  batch=512, shuffle=False, augment=False)
train_ds_aug   = make_dataset(trainX, trainY, batch=512, shuffle=True,  augment=True, p=0.5)
