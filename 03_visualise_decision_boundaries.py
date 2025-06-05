import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = np.load("svm_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

svm_linear = SVC(kernel='linear', C=1.0).fit(X_train, y_train)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)

# meshgrid for plotting
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

def plot_decision_boundary(model, title, subplot):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    subplot.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)
    subplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolor='k')
    subplot.set_title(title)
    subplot.set_xlabel("Feature 1")
    subplot.set_ylabel("Feature 2")

# plotting both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(svm_linear, "Linear SVM", ax1)
plot_decision_boundary(svm_rbf, "RBF SVM", ax2)
plt.tight_layout()
plt.show()
