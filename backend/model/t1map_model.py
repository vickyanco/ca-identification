import tensorflow as tf
from tensorflow.keras import layers, models, regularizers # type: ignore
from .base_model import BaseModel

class T1Model(BaseModel):
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling3D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model