# file: train.py
# description: Script to train LGE and T1 models
# author: María Victoria Anconetani
# date: 12/02/2025

from preprocessing.lge_preprocessor import LGEPreprocessor
from preprocessing.t1map_preprocessor import T1Preprocessor
from utils.data_loader_augmentation import DataLoader
from model.lge_model import LGEModel
from model.t1map_model import T1Model
import tensorflow as tf
# Paths to Data
lge_cases_path = "E:/CA EN CMR/lge-casos-nifti"
lge_controls_path = "E:/CA EN CMR/lge-controles-nifti"
t1_cases_path = "E:/CA EN CMR/t1_mapping_corregido/casos/"
t1_controls_path = "E:/CA EN CMR/t1_mapping_corregido/controles/"

# Instantiate preprocessors with target shape
lge_preprocessor = LGEPreprocessor(target_shape=(172, 192, 12))
t1_preprocessor = T1Preprocessor(target_shape=(256, 218, 8))

# Load Data
data_loader = DataLoader(lge_cases_path, lge_controls_path, t1_cases_path, t1_controls_path, 
                        lge_preprocessor, t1_preprocessor)

(lge_X, lge_y, lge_class_weights), (t1_X, t1_y, t1_class_weights) = data_loader.load_all_data()

# Convert to Tensors
lge_X, lge_y = tf.convert_to_tensor(lge_X, dtype=tf.float32), tf.convert_to_tensor(lge_y, dtype=tf.int32)
t1_X, t1_y = tf.convert_to_tensor(t1_X, dtype=tf.float32), tf.convert_to_tensor(t1_y, dtype=tf.int32)

# Split into train and validation sets
split_lge = int(0.8 * len(lge_X))
split_t1 = int(0.8 * len(t1_X))

lge_train_data = (lge_X[:split_lge], lge_y[:split_lge])
lge_val_data = (lge_X[split_lge:], lge_y[split_lge:])

t1_train_data = (t1_X[:split_t1], t1_y[:split_t1])
t1_val_data = (t1_X[split_t1:], t1_y[split_t1:])

# Convert class weights to dictionary format
lge_class_weights = {0: lge_class_weights[0], 1: lge_class_weights[1]}
t1_class_weights = {0: t1_class_weights[0], 1: t1_class_weights[1]}

# Create models
lge_model = LGEModel()
t1_model = T1Model()

# Debugging: Check Data Shapes
print(f"LGE Training Data Shape: {lge_train_data[0].shape}, Labels: {lge_train_data[1].shape}")
print(f"T1 Training Data Shape: {t1_train_data[0].shape}, Labels: {t1_train_data[1].shape}")

# Ensure y_train is not None
if lge_train_data[1] is None or t1_train_data[1] is None:
    raise ValueError("🚨 Labels (y_train) are None. Check DataLoader output!")

# Train models
lge_model.train(lge_train_data, epochs=10, class_weight=lge_class_weights)
t1_model.train(t1_train_data, epochs=10, class_weight=t1_class_weights)

# Evaluate models
print("\n🔎 Evaluating LGE Model:")
lge_model.evaluate(lge_val_data[0], lge_val_data[1])

print("\n🔎 Evaluating T1 Model:")
t1_model.evaluate(t1_val_data[0], t1_val_data[1])
