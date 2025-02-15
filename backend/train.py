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

# Instantiate preprocessor and data loader
lge_preprocessor = LGEPreprocessor()
t1_preprocessor = T1Preprocessor()
# Load Data
data_loader = DataLoader(lge_cases_path, lge_controls_path, t1_cases_path, t1_controls_path, 
                        lge_preprocessor, t1_preprocessor)

(lge_X, lge_y, lge_class_weights), (t1_X, t1_y, t1_class_weights) = data_loader.load_all_data()

# Convert to Tensors
lge_X, lge_y = tf.convert_to_tensor(lge_X, dtype=tf.float32), tf.convert_to_tensor(lge_y, dtype=tf.int32)
t1_X, t1_y = tf.convert_to_tensor(t1_X, dtype=tf.float32), tf.convert_to_tensor(t1_y, dtype=tf.int32)

# Create Models

# Create models
lge_model = LGEModel(input_shape=(128, 128, 128, 1))
t1_model = T1Model(input_shape=(128, 128, 128, 1))

# Split into train and validation sets
split = int(0.8 * len(lge_X))  # 80% training, 20% validation
lge_train_data = (lge_X[:split], lge_y[:split])
lge_val_data = (lge_X[split:], lge_y[split:])

t1_train_data = (t1_X[:split], t1_y[:split])
t1_val_data = (t1_X[split:], t1_y[split:])

# Train models with class weights
lge_model.train(lge_train_data, validation_data=lge_val_data, epochs=10)
t1_model.train(t1_train_data, validation_data=t1_val_data, epochs=10)
