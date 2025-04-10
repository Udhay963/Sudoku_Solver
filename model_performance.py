import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import random

# Load model and test dataset
model = load_model("sudoku_digit_model.h5")
data = np.load("sudoku_digits.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Preprocess test data
X_test = X_test.reshape(-1, 28, 28, 1)
y_test_cat = to_categorical(y_test, 10)

# Evaluate model to get accuracy and loss history
# NOTE: If you didn't save `history` during training, comment the below line
# history = model.fit(...) → Save history during training to reuse

# Predict class probabilities
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# ---------------------- 1. CONFUSION MATRIX ----------------------
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Digit Recognition")
plt.grid(False)
plt.show()

# ---------------------- 2. SAMPLE PREDICTIONS ----------------------
plt.figure(figsize=(10, 10))
indices = random.sample(range(len(X_test)), 9)
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("Sample Predictions (Actual vs. Predicted)", fontsize=16)
plt.tight_layout()
plt.show()