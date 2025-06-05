# **SVM Linear vs RBF Classification**  
This project demonstrates Support Vector Machines (SVM) for binary classification on a synthetic 2D dataset generated with `make_moons`. It compares linear and RBF kernels, covering data preparation, training, hyperparameter tuning, cross-validation, and decision boundary visualization.

📌 **Project Overview**  
The dataset contains 2D points forming two interleaving half circles (moons), a classic example of a non-linearly separable problem. The project shows how linear and RBF kernel SVMs perform on this data, including scaling features, tuning `C` and `gamma`, and evaluating results with cross-validation.

✅ **Key Steps Performed**  
📥 Data Generation & Preparation  
Generated the `make_moons` dataset, visualized it, scaled features using StandardScaler, and split into training and testing sets.

⚙️ Training Linear & RBF SVM Models  
Trained two SVM classifiers with linear and RBF kernels and evaluated accuracy on test data.

🖼️ Decision Boundary Visualization  
Plotted decision boundaries for both linear and RBF SVM models in 2D feature space.

🔍 Hyperparameter Tuning  
Used GridSearchCV to find the best `C` and `gamma` parameters for the RBF kernel.

📊 Cross-Validation Evaluation  
Performed 5-fold cross-validation on the tuned RBF model to estimate generalization performance.

🛠️ **Tools & Libraries Used**  
- Python 3  
- NumPy – Numerical operations  
- Scikit-learn – Dataset generation, preprocessing, SVM modeling, hyperparameter tuning, cross-validation  
- Matplotlib – Plotting data and decision boundaries  

📁 **Project Structure**  
svm-linear-vs-rbf/  
├── 01_load_prepare_dataset.py         # Generate, scale, split, and save dataset  
├── 02_train_svm_models.py             # Train linear and RBF SVM, report accuracy  
├── 03_visualize_decision_boundaries.py # Visualize decision boundaries of both models  
├── 04_tune_hyperparameters.py         # GridSearchCV for `C` and `gamma` parameters  
├── 05_cross_validation_evaluation.py  # Cross-validation on tuned RBF SVM model  
└── README.md                         # Project documentation  

🎯 **Goal**  
To understand and compare SVM classifiers with linear and RBF kernels on a non-linear dataset, learning data preprocessing, model training, tuning, evaluation, and visualization.

🙌 **Acknowledgements**  
Dataset source: Synthetic `make_moons` dataset from Scikit-learn

📬 **Contact**  
Sanskriti Anya  
📧 sanskritianya17@gmail.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sanskriti-anya-6bb2b4332)
