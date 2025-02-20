# file: model/lge_model.py
# description: Model for LGE Images
# author: María Victoria Anconetani
# date: 12/02/2025

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore

class LGE_CNN:
    def __init__(self, input_shape=(172, 192, 12, 1), num_classes=1, dropout_rate=0.3, l2_reg=1e-4, initial_lr=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.initial_lr = initial_lr
        
        self.model = self.build_model()
    
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),

            # 1st Conv Block
            layers.Conv3D(32, (3, 3, 3), padding='same', strides=(1, 1, 1), activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),

            layers.Conv3D(64, (3, 3, 3), padding='same', strides=(2, 2, 1), activation='relu',  # <-- Prevents Z reduction
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),

            # 2nd Conv Block
            layers.Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1), activation='relu',
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),

            layers.Conv3D(256, (3, 3, 3), padding='same', strides=(2, 2, 1), activation='relu',  # <-- Prevents Z reduction
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),

            # Global Feature Extraction
            layers.GlobalAveragePooling3D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        return model

    def summary(self):
        """Prints the model summary."""
        self.model.summary()

    def get_callbacks(self):
        """Returns a list of callbacks for training."""
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        return [lr_callback]

model = LGE_CNN()
model.summary()