from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"   
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


print(f"BASE_DIR = {BASE_DIR}")
print(f"DATA_DIR = {DATA_DIR}")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def prepare_dataloader(ds: tf.data.Dataset, training: bool = False) -> tf.data.Dataset:
    if training:
        ds = ds.shuffle(1000, seed=SEED)

    ds = (
        ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
          .prefetch(AUTOTUNE)
    )
    return ds



def load_rps_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names}")

    print("Tamaño batch de train:", train_ds.cardinality().numpy())
    print("Tamaño batch de val:", val_ds.cardinality().numpy())

    return train_ds, val_ds, class_names



def build_cnn_model(input_shape=(160, 160, 3)) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax"), 
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

def train_model(model, train_ds, val_ds, epochs=5):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def evaluate_model(model, val_ds):
    loss, acc = model.evaluate(val_ds)
    print(f"\nResultados validación -> loss: {loss:.4f} | acc: {acc:.4f}")



def save_model(model, name: str = "cnn_rps.keras") -> Path:
    if not (name.endswith(".keras") or name.endswith(".h5")):
        name = name + ".keras"

    export_path = MODELS_DIR / name
    model.save(export_path)
    print(f"Modelo guardado en: {export_path.resolve()}")
    return export_path


def print_model_limitations():
    print("\n=== Limitaciones del sistema de reconocimiento ===")
    print("- Solo reconoce tres gestos: 'rock', 'paper' y 'scissors'.")
    print("- No está diseñado para otros gestos ni poses de manos.")
    print("- Puede fallar si la imagen está muy oscura, borrosa o la mano")
    print("  está muy lejos/cerca de la cámara.")
    print("- El modelo fue entrenado con imágenes de este dataset; en otros")
    print("  contextos el rendimiento puede ser menor.")
    print("=================================================\n")



def main():
    train_raw, val_raw, class_names = load_rps_datasets()

    train_ds = prepare_dataloader(train_raw, training=True)
    val_ds = prepare_dataloader(val_raw, training=False)

    model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    history = train_model(model, train_ds, val_ds, epochs=5)

    evaluate_model(model, val_ds)

    save_model(model, name="cnn_rps")

    print_model_limitations()


if __name__ == "__main__":
    main()