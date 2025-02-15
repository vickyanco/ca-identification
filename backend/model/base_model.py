# file: base_model.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
class BaseModel:
    def __init__(self, input_shape, num_classes=2, dropout_rate=0.2, l2_reg=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = self.build_model()
    
    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model method")
    
    def train(self, train_data, validation_data, epochs=10, batch_size=8, callbacks=[]):
        # Calcular pesos de clase para balanceo de datos
        y_train = train_data[1].numpy()
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        return self.model.fit(train_data[0], train_data[1],
                                epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data,
                                class_weight=class_weight_dict,
                                callbacks=callbacks)
    
    def evaluate(self, test_data):
        return self.model.evaluate(test_data)