def load_rml2016a(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d.keys()})
    mod2idx = {m:i for i,m in enumerate(mods)}
    Xs, Ys, Ss = [], [], []
    for (mod, snr), frames in d.items():   # frames: [N, 2, 128]
        frames = frames.astype(np.float32)
        Xs.append(frames)
        Ys.append(np.full(frames.shape[0], mod2idx[mod], np.int32))
        Ss.append(np.full(frames.shape[0], snr, np.int32))
    X  = np.concatenate(Xs, 0)   # [N,2,128]
    y  = np.concatenate(Ys, 0)   # [N]
    sn = np.concatenate(Ss, 0)   # [N]
    return X, y, sn, mods

X, y_mod, y_snr, MODS = load_rml2016a(PKL_PATH)
print("Original X shape:", X.shape, "classes:", len(MODS), MODS[:5], "...")

USE_FRAME_STACKING = False  # set True for better accuracy (slightly slower)
if USE_FRAME_STACKING:
    def stack_adjacent_frames(X, every=2):
        N = (X.shape[0] // every) * every
        X2 = X[:N].reshape(-1, every, 2, X.shape[-1])        # [N/2,2,2,128]
        X2 = np.concatenate([X2[:,0], X2[:,1]], axis=-1)     # [N/2,2,256]
        return X2
    X = stack_adjacent_frames(X, every=2)
    y_mod = y_mod[:X.shape[0]]
    y_snr = y_snr[:X.shape[0]]
INPUT_LEN = X.shape[-1]
print("Final X shape:", X.shape, "| INPUT_LEN =", INPUT_LEN)

def make_splits(X, y, snr, test_size=0.2, val_size=0.1, seed=42):
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, snr, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva, str_, sva = train_test_split(
        Xtr, ytr, str_, test_size=val_size, stratify=ytr, random_state=seed)
    return (Xtr,ytr,str_), (Xva,yva,sva), (Xte,yte,ste)

(trainX, trainY, trainS), (valX, valY, valS), (testX, testY, testS) = make_splits(X, y_mod, y_snr)
print("Train/Val/Test:", trainX.shape, valX.shape, testX.shape)

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
