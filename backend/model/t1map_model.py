# file: model/t1map_model.py
# description: Model for T1 Mapping 
# author: María Victoria Anconetani
# date: 12/02/2025

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers # type: ignore

class DeepT1Model:
    def __init__(self, input_shape=(256, 218, 8, 1), num_classes=2, dropout_rate=0.3, l2_reg=1e-4, initial_lr=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.initial_lr = initial_lr
        
        self.model = self.build_model()
    
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First Conv Block
            layers.Conv3D(32, (3, 3, 3), padding='same', strides=(1, 1, 1),
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv3D(64, (3, 3, 3), padding='same', strides=(2, 2, 2),
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Second Conv Block
            layers.Conv3D(128, (3, 3, 3), padding='same', strides=(1, 1, 1),
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv3D(256, (3, 3, 3), padding='same', strides=(2, 2, 2),
                        kernel_regularizer=regularizers.l2(self.l2_reg)),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Global Feature Extraction
            layers.GlobalAveragePooling3D(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Adaptive Learning Rate Scheduler
        lr_schedule = tf.keras.optimizers.schedules.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, train_data, validation_data, epochs=15, batch_size=8, callbacks=[]):
        """Train the model with class weighting to handle imbalance."""
        
        # Extract labels safely
        y_train = np.concatenate([labels.numpy() for _, labels in train_data])
        
        # Convert one-hot to single-label if necessary
        if len(y_train.shape) > 1 and y_train.shape[1] == 2:
            y_train = np.argmax(y_train, axis=1)

        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        print("Computed Class Weights:", class_weight_dict)

        history = self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        return history
    
    def evaluate(self, test_data):
        """Evaluate the model with all required metrics."""
        results = self.model.evaluate(test_data, verbose=1)  
        loss, accuracy = results[0], results[1]  

        y_true, y_pred_probs = [], []

        for images, labels in test_data:
            y_true.append(labels.numpy())  
            y_pred_probs.append(self.model.predict(images, verbose=0))  

        # Convert to NumPy arrays and flatten
        y_true = np.concatenate(y_true, axis=0)
        y_pred_probs = np.concatenate(y_pred_probs, axis=0).flatten()

        # Convert probabilities to binary labels
        y_pred_labels = (y_pred_probs > 0.5).astype(int)

        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred_probs)
        metrics["loss"] = float(loss)
        metrics["accuracy"] = float(accuracy)

        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")

        return metrics

    def compute_metrics(self, y_true, y_pred_probs):
        """Compute evaluation metrics for binary classification."""
        y_pred_labels = (y_pred_probs > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred_labels)
        print("\n🔹 Confusion Matrix:\n", cm)

        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0  

        accuracy = accuracy_score(y_true, y_pred_labels)
        precision = precision_score(y_true, y_pred_labels, zero_division=0)
        recall = recall_score(y_true, y_pred_labels, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall (sensitivity)": recall,
            "specificity": specificity,
            "f1_score": f1,
            "auc": auc
        }
