import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from backend.preprocessing.load_data import LGEDataLoader

# Load dataset
dataset_root = "E:/CA EN CMR/LGE_prep_nii_divided"
data_loader = LGEDataLoader(dataset_root)
data_loader.prepare_datasets()

test_dataset = data_loader.test_dataset

# Load trained model
model = tf.keras.models.load_model("lge_cnn_model.h5")

# Evaluate model
y_true, y_pred_probs = [], []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images, verbose=0).flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred_labels = (y_pred_probs > 0.5).astype(int)

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred_labels)
precision = precision_score(y_true, y_pred_labels, zero_division=0)
recall = recall_score(y_true, y_pred_labels, zero_division=0)
auc_score = roc_auc_score(y_true, y_pred_probs)
cm = confusion_matrix(y_true, y_pred_labels)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = f1_score(y_true, y_pred_labels)

# Print Results
print(f"\n🔹 Test Results:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall (Sensitivity): {recall:.4f}")
print(f"✅ Specificity: {specificity:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ ROC-AUC: {auc_score:.4f}")
print("\n🔹 Confusion Matrix:")
print(cm)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
