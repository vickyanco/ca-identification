# file: utils/plots.py
# description: 
# author: María Victoria Anconetani
# date: 21/03/2025

from tensorflow.keras.models import load_model # type: ignore
from graphviz import Digraph

# Load models from .h5 files
t1_model = load_model("t1_mapping_cnn_model.h5")  # Replace with actual path
lge_model = load_model("lge_cnn_model.h5")  # Replace with actual path

def plot_model_graph(model, filename):
    graph = Digraph(comment="Model Architecture", format="png")
    
    for i, layer in enumerate(model.layers):
        try:
            # Intentar obtener el `output_shape` de forma segura
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "N/A"
            graph.node(f"L{i}", label=f"{layer.name}\n{output_shape}", shape="box")
        except AttributeError:
            # Manejar cualquier error de acceso
            graph.node(f"L{i}", label=f"{layer.name}\nSin forma", shape="box")
    
    for i in range(len(model.layers) - 1):
        graph.edge(f"L{i}", f"L{i+1}")

    graph.render(filename, view=True)

plot_model_graph(t1_model, "t1_model_grafo")
plot_model_graph(lge_model, "lge_model_grafo")
