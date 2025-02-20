# file: train/train_lge.py
# description: Script to train the LGE model.
# author: María Victoria Anconetani
# date: 20/02/2025

from backend.preprocessing.load_data import LGEDataLoader
from backend.model.lge_model import LGE_CNN

# Load dataset
dataset_root = "E:/CA EN CMR/LGE_prep_nii_divided" 
data_loader = LGEDataLoader(dataset_root)
data_loader.prepare_datasets()

train_dataset = data_loader.train_dataset
test_dataset = data_loader.test_dataset

# Initialize model
model = LGE_CNN()
callbacks = model.get_callbacks()

# Train model
history = model.model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,  # Adjust as needed
    callbacks=callbacks
)

# Save trained model
model.model.save("lge_cnn_model.h5")
print("✅ Model Trained & Saved Successfully")
