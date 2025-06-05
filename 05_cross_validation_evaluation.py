import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data = np.load("svm_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# final model with best parameters
final_model = SVC(kernel='rbf', C=100, gamma=1)

# performing 5-fold cross-validation
cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')

print(" Cross-Validation Scores:", np.round(cv_scores, 4))
print(" Mean CV Accuracy:", round(np.mean(cv_scores), 4))
print(" Standard Deviation:", round(np.std(cv_scores), 4))
