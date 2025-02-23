# file: train/train_t1.py
# description: Script to train the T1 Mapping model.
# author: María Victoria Anconetani
# date: 22/02/2025

from backend.preprocessing.load_t1_data import T1DataLoader  
from backend.model.t1map_model import T1MappingCNN  
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore 
import numpy as np
import tensorflow as tf
import os

# Load dataset
dataset_root = r"E:/CA EN CMR/T1Map_prepro/"  
data_loader = T1DataLoader(dataset_root, batch_size=16)
data_loader.prepare_datasets()

train_dataset = data_loader.train_dataset
test_dataset = data_loader.test_dataset

# Extract labels from the training dataset
y_train = np.concatenate([labels for _, labels in train_dataset.as_numpy_iterator()])
unique, counts = np.unique(y_train, return_counts=True)
unique, counts = np.unique(y_train, return_counts=True)
print("✅ Unique Classes in y_train:", unique)
print("✅ Class Counts:", counts)

# Compute class weights safely
unique_classes = np.unique(y_train)
print("✅ Unique Classes in y_train:", unique_classes)
print("✅ Class Counts:", np.bincount(y_train))  # Show count per class

if len(unique_classes) < 2:
    print("🚨 WARNING: Only one class found in y_train! Defaulting to equal class weights.")
    class_weight_dict = {0: 1.0, 1: 1.0}  # Set equal weights
else:
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    print("✅ Raw Class Weights:", class_weights)  # Debugging line

    class_weight_dict = {
        0: class_weights[0] * 1.0,  
        1: class_weights[1] * 1.40   
    }

print("✅ Adjusted Class Weights:", class_weight_dict)

# Initialize 2D CNN model
model = T1MappingCNN()

# Compile model
model.model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Adjust learning rate if needed
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Add EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Train model
history = model.model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100,  
    class_weight=class_weight_dict,  
    callbacks=[early_stopping]
)

# Save trained model
model.model.save("t1_mapping_cnn_model.h5")
print("✅ Model Trained & Saved Successfully")