# file: utils/plots.py
# description: 
# author: María Victoria Anconetani
# date: 21/03/2025

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore

# Load models from .h5 files
t1_model = load_model("t1_mapping_cnn_model.h5")  # Replace with actual path
lge_model = load_model("lge_cnn_model.h5")  # Replace with actual path

# Now plot the model
plot_model(t1_model, to_file="t1_model.png", show_shapes=True, show_layer_names=True)
plot_model(lge_model, to_file="lge_model.png", show_shapes=True, show_layer_names=True)