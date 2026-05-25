import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load dataset (MNIST for simplicity)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model function (same architecture)
def create_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Optimizers to compare
optimizers = {
    "SGD": keras.optimizers.SGD(),
    "RMSprop": keras.optimizers.RMSprop(),
    "Adam": keras.optimizers.Adam()
}

histories = {}

# Train model with each optimizer
for name, optimizer in optimizers.items():
    print(f"\nTraining with {name} optimizer...\n")

    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    histories[name] = history

# Plot accuracy curves
plt.figure(figsize=(12, 6))

for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=f"{name} - Val Acc")
    plt.plot(history.history['accuracy'], linestyle='--', label=f"{name} - Train Acc")

plt.title("Optimizer Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Compare final results
print("\nFinal Validation Accuracy:")
for name, history in histories.items():
    final_acc = history.history['val_accuracy'][-1]
    print(f"{name}: {final_acc:.4f}")