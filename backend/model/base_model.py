# file: base_model.py
# description: Base class for image preprocessing
# author: María Victoria Anconetani
# date: 12/02/2025

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

class BaseModel:
    def __init__(self, input_shape, learning_rate=1e-5, l2_reg=0.001, dropout_rate=0.3):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError("Subclasses must implement `build_model()`")

    def train(self, dataset, epochs=10, class_weight=None):
        """
        Train the model using the provided dataset and class weights.
        """
        
        X_train, y_train = dataset  # Extract features and labels properly

        history = self.model.fit(X_train, y_train, epochs=epochs, class_weight=class_weight)

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and print classification metrics + ROC Curve.
        """
        y_pred_probs = self.model.predict(X_test).flatten()
        y_pred = (y_pred_probs >= 0.5).astype(int)  # Convert probabilities to binary predictions
        
        # 🏆 Print Classification Metrics
        print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, digits=4))

        # 🏆 Compute ROC Curve & AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # 🔥 Plot ROC Curve
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()