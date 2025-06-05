import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = np.load("svm_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# defining parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto']
}

# grid search with cross-validation
grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf'),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# best parameters and score
print(" Best Parameters:", grid_search.best_params_)
print(" Best Cross-Validation Accuracy:", round(grid_search.best_score_, 4))
