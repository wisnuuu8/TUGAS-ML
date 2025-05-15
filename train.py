from getData import train_ds, val_ds
from model import create_disaster_cnn
from loss_accuracy import LossAccuracyPlotter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, val_ds):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    class_names = val_ds.class_names

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

def main():
    # Cek apakah data tersedia
    if train_ds is None or val_ds is None:
        raise ValueError("train_ds atau val_ds tidak boleh None. Periksa kembali getData.py.")

    # Buat model
    model = create_disaster_cnn()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Latih model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    # Plot loss & accuracy
    plotter = LossAccuracyPlotter(history)
    plotter.plot()

    # Plot confusion matrix
    plot_confusion_matrix(model, val_ds)

    # Simpan model
    model.save("disaster_cnn_model.h5")
    print("✅ Model disimpan ke disaster_cnn_model.h5")

if __name__ == "__main__":
    main()
