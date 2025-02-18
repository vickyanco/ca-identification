# file: model/t1map_model.py
# description: Model for T1 Mapping 
# author: María Victoria Anconetani
# date: 12/02/2025

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from .base_model import BaseModel

class T1Model(BaseModel):
    def __init__(self):
        super().__init__(input_shape=(256, 218, 8, 1))  # Updated shape

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),  # Explicit Input Layer
            layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu', 
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),

            layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu', 
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),

            layers.GlobalAveragePooling3D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        return model