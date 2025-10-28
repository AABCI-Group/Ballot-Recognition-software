# tools/train_punch_classifier.py
import os, tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "../data_punch"
IMG_SIZE = 160
BATCH = 32
EPOCHS = 15
OUT = "../models/punchclf_v1.keras"

def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    # lightweight backbone
    base = tf.keras.applications.MobileNetV3Small(
        input_tensor=inp, include_top=False, weights=None, alpha=0.75
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def make_ds(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels="inferred", label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, shuffle=True
    )
    # strong but safe augs
    aug = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.1),
    ])
    return ds.map(lambda x,y: (aug(x, training=True), y)).prefetch(2)

def main():
    tr = make_ds("train")
    va = make_ds("val")
    model = build_model()
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    ]
    model.fit(tr, validation_data=va, epochs=EPOCHS, callbacks=cb)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model.save(OUT)
    print(f"[OK] Saved {OUT}")

if __name__ == "__main__":
    main()
