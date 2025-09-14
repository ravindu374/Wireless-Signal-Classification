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

optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)  # start conservative

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("checkpoints/best_resnet.keras", save_best_only=True, monitor='val_accuracy')
]

# 2–3 epoch warm-up without augmentation (stabilizes early learning)
print("Warm-up (no augmentation)...")
model.fit(train_ds_noaug, validation_data=val_ds, epochs=3, verbose=1)

# Main training with probabilistic augmentation
print("Main training (with augmentation p=0.5)...")
history = model.fit(train_ds_aug, validation_data=val_ds, epochs=100, callbacks=callbacks, verbose=1)

print("\nEvaluate on TRAIN (no augmentation):")
train_eval_ds = make_dataset(trainX, trainY, batch=512, shuffle=False, augment=False)
model.evaluate(train_eval_ds, verbose=1)

print("Evaluate on VALIDATION:")
model.evaluate(val_ds, verbose=1)

print("Evaluate on TEST:")
model.evaluate(test_ds, verbose=1)

# Predict on test for acc vs SNR
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)

# Rebuild test indices to align preds→labels
# (Because we batched from arrays directly, order is preserved.)
import matplotlib.pyplot as plt

snrs = testS
xs = sorted(list(set(snrs.tolist())))
ys = []
for s in xs:
    m = (snrs == s)
    ys.append(float((testY[m] == y_pred[m]).mean()))

plt.figure(figsize=(6,4))
plt.plot(xs, ys, marker='o')
plt.title("Accuracy vs SNR (Test)")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
