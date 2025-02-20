# file: train/train_lge.py
# description: Script to train the LGE model.
# author: María Victoria Anconetani
# date: 20/02/2025

from sklearn.utils.class_weight import compute_class_weight
from backend.preprocessing.load_data import LGEDataLoader
from backend.model.lge_model import LGE_CNN
import numpy as np

# Load dataset
dataset_root = "E:/CA EN CMR/LGE_prep_nii_divided" 
data_loader = LGEDataLoader(dataset_root)
data_loader.prepare_datasets()

train_dataset = data_loader.train_dataset
test_dataset = data_loader.test_dataset

# Extract labels from the training dataset
y_train = np.concatenate([labels.numpy() for _, labels in train_dataset])
unique, counts = np.unique(y_train, return_counts=True)
print("✅ Class Distribution:", dict(zip(unique, counts)))

# Compute class weights based on training data
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print("✅ Computed Class Weights:", class_weight_dict)

# Initialize model
model = LGE_CNN()
callbacks = model.get_callbacks()

# Train model
history = model.model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,  
    class_weight=class_weight_dict,  
    callbacks=callbacks
)

# Save trained model
model.model.save("lge_cnn_model.h5")
print("✅ Model Trained & Saved Successfully")
