# file: model/t1map_model.py
# description: Model for T1 Mapping 
# author: María Victoria Anconetani
# date: 12/02/2025

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore

class T1MappingCNN:
    def __init__(self, input_shape=(256, 256, 1), num_classes=1, dropout_rate=0.4, l2_reg=1e-4, initial_lr=1e-4):
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
            layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.MaxPooling2D((2,2)),

            # 2nd Conv Block
            layers.Conv2D(64, (3,3), activation='relu', padding='same', 
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.MaxPooling2D((2,2)),

            # 3rd Conv Block 
            layers.Conv2D(128, (3,3), activation='relu', padding='same', 
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.MaxPooling2D((2,2)),

            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),

            # Fully Connected Layer
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.Dropout(self.dropout_rate),

            # Output Layer
            layers.Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def summary(self):
        """Prints the model summary."""
        self.model.summary()
    
    def get_callbacks(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1)

        return [early_stopping, reduce_lr]

model = T1MappingCNN()
model.summary()
