import json

nb = json.load(open('HW8/Part_B/HW08-PartB_Adam_Nelson-Archer.ipynb'))

for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        
        # 1. Update data loading to include Central Crop
        if 'def decode_and_resize' in source and 'def make_autoencoder_ds' in source:
            new_source = """
def decode_and_resize(path: tf.Tensor, img_size: tuple[int, int]):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    # NEW: Center crop to remove 25% of the background edges!
    img = tf.image.central_crop(img, central_fraction=0.75)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((img_size[0], img_size[1], 3))
    return img

def make_autoencoder_ds(paths: list[Path], img_size: tuple[int, int], batch_size: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    ds = ds.map(lambda p: decode_and_resize(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    # CACHE THE DECODED IMAGES IN RAM! No more disk reads after epoch 1.
    ds = ds.cache() 
    
    if shuffle:
        ds = ds.shuffle(buffer_size=max(1000, len(paths)), seed=SEED, reshuffle_each_iteration=True)
        
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_autoencoder_ds(normal_train_paths, cfg_img_size, cfg_batch_size, shuffle=True)
val_ds = make_autoencoder_ds(normal_val_paths, cfg_img_size, cfg_batch_size, shuffle=False)

print(train_ds)
print(val_ds)
"""
            c['source'] = new_source.splitlines(True)
            continue
        
        # 2. Revert Model Definition
        if 'def build_conv_autoencoder_like_reference' in source and 'ssim' in source.lower():
            new_source = """
def build_conv_autoencoder_like_reference(img_size: tuple[int, int]):
    inp = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image")

    # GPU-accelerated Data Augmentation
    aug = tf.keras.layers.RandomFlip("horizontal")(inp)
    aug = tf.keras.layers.RandomRotation(0.1)(aug)
    aug = tf.keras.layers.RandomZoom(0.1)(aug)

    # Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(aug)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool3")(x)
    # Shape is 12x12x16 = 2304

    # Dense Bottleneck for better KDE performance
    x_flat = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(64, activation="relu", name="encoded")(x_flat)

    # Decoder
    x = tf.keras.layers.Dense(12 * 12 * 16, activation="relu")(encoded)
    x = tf.keras.layers.Reshape((12, 12, 16))(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="dec_conv1")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up1")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up2")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv3")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up3")(x)

    out = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inp, out, name="conv_autoencoder")
    encoder = tf.keras.Model(inp, encoded, name="encoder_replica")
    return autoencoder, encoder

autoencoder, encoder = build_conv_autoencoder_like_reference(cfg_img_size)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mae",
)

print("Autoencoder summary:")
autoencoder.summary()
print("\\nEncoder summary (compressed to 64):")
encoder.summary()
"""
            c['source'] = new_source.splitlines(True)
            continue
            
        # Update plotting labels from SSIM Error to MAE
        if 'Reconstruction SSIM Error Distribution by Flower Type' in source:
            c['source'] = [line.replace('SSIM Error', 'MAE').replace('ssim_error', 'mae').replace('ssim', 'mae') for line in c['source']]
            
        # Update reconstruction_ssim to reconstruction_mae and add Central Crop here too
        if 'def reconstruction_ssim' in source:
            new_source = """
def load_array(paths: list[Path], img_size: tuple[int, int]) -> np.ndarray:
    arr = []
    for p in paths:
        b = tf.io.read_file(str(p))
        img = tf.image.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.central_crop(img, central_fraction=0.75) # Match dataset
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        arr.append(img.numpy())
    return np.asarray(arr, dtype=np.float32)

def reconstruction_mae(model: tf.keras.Model, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.abs(x - recon), axis=(1, 2, 3))

# Normal train/validation/test arrays
x_train = load_array(normal_train_paths, cfg_img_size)
x_val = load_array(normal_val_paths, cfg_img_size)
x_norm_test = load_array(normal_test_paths, cfg_img_size)

# Per-class anomaly arrays
x_anomaly_by_class: dict[str, np.ndarray] = {}
for cls in anomaly_classes:
    x_anomaly_by_class[cls] = load_array(anomaly_test_paths[cls], cfg_img_size)

# Aggregate anomaly array (for reference-style error print)
x_anomaly_all = np.concatenate([x_anomaly_by_class[c] for c in anomaly_classes], axis=0)

val_error_mean = float(np.mean(reconstruction_mae(autoencoder, x_val)))
anomaly_error_mean = float(np.mean(reconstruction_mae(autoencoder, x_anomaly_all)))
print(f"Error on validation set: {val_error_mean:.6f}, error on anomaly set: {anomaly_error_mean:.6f}")

# Requirement comparison by flower type
mae_by_class: dict[str, np.ndarray] = {cfg_normal_class: reconstruction_mae(autoencoder, x_norm_test)}
for cls in anomaly_classes:
    mae_by_class[cls] = reconstruction_mae(autoencoder, x_anomaly_by_class[cls])

summary_rows = []
for cls, values in mae_by_class.items():
    summary_rows.append(
        {
            "class": cls,
            "mean_mae": float(np.mean(values)),
            "std_mae": float(np.std(values)),
            "median_mae": float(np.median(values)),
            "n": len(values),
        }
    )

import pandas as pd
mae_summary_df = pd.DataFrame(summary_rows).sort_values("mean_mae")
mae_summary_df
"""
            c['source'] = new_source.splitlines(True)
            continue
            
        if 'plt.plot(history.history["loss"]' in source and 'SSIM Error' in source:
             c['source'] = [line.replace('SSIM Error', 'MAE') for line in c['source']]

with open('HW8/Part_B/HW08-PartB_Adam_Nelson-Archer.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
