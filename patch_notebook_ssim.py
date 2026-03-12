import json

nb = json.load(open('HW8/Part_B/HW08-PartB_Adam_Nelson-Archer.ipynb'))

for c in nb['cells']:
    if c.get('cell_type') == 'code':
        source = "".join(c.get('source', []))
        
        # 1. Update Model Definition with SSIM loss and larger bottleneck
        if 'def build_conv_autoencoder_like_reference' in source:
            new_source = """
@tf.keras.utils.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def build_conv_autoencoder_like_reference(img_size: tuple[int, int]):
    inp = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="image")

    # GPU-accelerated Data Augmentation
    aug = tf.keras.layers.RandomFlip("horizontal")(inp)
    aug = tf.keras.layers.RandomRotation(0.1)(aug)
    aug = tf.keras.layers.RandomZoom(0.1)(aug)

    # Encoder (Deeper, more capacity)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv1")(aug)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same", name="enc_pool3")(x)
    # Shape is 12x12x16 = 2304

    # Dense Bottleneck: Increased to 256 for much better reconstruction
    x_flat = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(256, activation="relu", name="encoded")(x_flat)

    # Decoder
    x = tf.keras.layers.Dense(12 * 12 * 16, activation="relu")(encoded)
    x = tf.keras.layers.Reshape((12, 12, 16))(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="dec_conv1")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up1")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up2")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv3")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), name="dec_up3")(x)

    out = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="reconstruction")(x)

    autoencoder = tf.keras.Model(inp, out, name="conv_autoencoder")
    encoder = tf.keras.Model(inp, encoded, name="encoder_replica")
    return autoencoder, encoder

autoencoder, encoder = build_conv_autoencoder_like_reference(cfg_img_size)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=ssim_loss,
)

print("Autoencoder summary:")
autoencoder.summary()
print("\\nEncoder summary (compressed to 256):")
encoder.summary()
"""
            c['source'] = new_source.splitlines(True)
            continue
            
        # Update plotting labels from MAE to SSIM Error
        if 'Reconstruction MAE Distribution by Flower Type' in source:
            c['source'] = [line.replace('MAE', 'SSIM Error').replace('mae', 'ssim') for line in c['source']]
            
        # Update reconstruction_mae to reconstruction_ssim
        if 'def reconstruction_mae' in source:
            new_source = """
def load_array(paths: list[Path], img_size: tuple[int, int]) -> np.ndarray:
    arr = []
    for p in paths:
        b = tf.io.read_file(str(p))
        img = tf.image.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        arr.append(img.numpy())
    return np.asarray(arr, dtype=np.float32)

def reconstruction_ssim(model: tf.keras.Model, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    # SSIM returns a 1D tensor of SSIM values per image in batch. 
    # 1.0 is perfect reconstruction. We return (1 - SSIM) as the "error" metric.
    ssim_vals = tf.image.ssim(tf.convert_to_tensor(x), tf.convert_to_tensor(recon), max_val=1.0)
    return 1.0 - ssim_vals.numpy()

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

val_error_mean = float(np.mean(reconstruction_ssim(autoencoder, x_val)))
anomaly_error_mean = float(np.mean(reconstruction_ssim(autoencoder, x_anomaly_all)))
print(f"SSIM Error on validation set: {val_error_mean:.6f}, error on anomaly set: {anomaly_error_mean:.6f}")

# Requirement comparison by flower type
ssim_by_class: dict[str, np.ndarray] = {cfg_normal_class: reconstruction_ssim(autoencoder, x_norm_test)}
for cls in anomaly_classes:
    ssim_by_class[cls] = reconstruction_ssim(autoencoder, x_anomaly_by_class[cls])

summary_rows = []
for cls, values in ssim_by_class.items():
    summary_rows.append(
        {
            "class": cls,
            "mean_ssim_error": float(np.mean(values)),
            "std_ssim_error": float(np.std(values)),
            "median_ssim_error": float(np.median(values)),
            "n": len(values),
        }
    )

import pandas as pd
ssim_summary_df = pd.DataFrame(summary_rows).sort_values("mean_ssim_error")
ssim_summary_df
"""
            c['source'] = new_source.splitlines(True)
            continue
            
        if 'plt.plot(history.history["loss"]' in source:
             c['source'] = [line.replace('MAE', 'SSIM Error') for line in c['source']]

with open('HW8/Part_B/HW08-PartB_Adam_Nelson-Archer.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
