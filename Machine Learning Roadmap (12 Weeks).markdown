# Machine Learning Roadmap (12 Weeks)

Based on *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition) as the primary guide, with CampusX videos as supplementary material.

## Week 1: Introduction to Machine Learning
- **Book Chapter**: Chapter 1: The Machine Learning Landscape
- **Topics**: What is ML? Why use ML? Types of ML (supervised, unsupervised, reinforcement). Batch vs. online learning. Instance-based vs. model-based learning. Challenges (e.g., data mismatch, overfitting). Testing and validation.
- **Videos**:
  - Video 1: What is Machine Learning? (20:00)
  - Video 2: AI Vs ML Vs DL for Beginners (16:02)
  - Video 3: Types of Machine Learning for Beginners (27:42)
  - Video 4: Batch Machine Learning (11:28)
  - Video 5: Online Machine Learning (19:28)
  - Video 6: Instance-Based Vs Model-Based Learning (16:44)
  - Video 7: Challenges in Machine Learning (23:40)
- **Goals**: Grasp ML’s definition, categories, and pitfalls (e.g., data mismatch in Iris example). Understand validation basics.
- **Practice**: Explore Iris dataset in Jupyter Notebook (code from Chapter 1).

## Week 2: End-to-End ML Project
- **Book Chapter**: Chapter 2: End-to-End Machine Learning Project
- **Topics**: ML project lifecycle (MLDLC). Data collection (CSV, API). Exploratory Data Analysis (EDA). Feature engineering. Model training and evaluation.
- **Videos**:
  - Video 9: Machine Learning Development Life Cycle (25:13)
  - Video 12: Installing Anaconda For Data Science (37:06)
  - Video 13: End to End Toy Project (30:43)
  - Video 15: Working with CSV files (36:30)
  - Video 19: Understanding Your Data (15:23)
  - Video 20: EDA using Univariate Analysis (30:31)
- **Goals**: Learn ML workflow, set up tools (Anaconda, Jupyter), and perform EDA. Apply to your GDP-Happiness dataset.
- **Practice**: Run Chapter 2’s housing project code. Merge and clean your GDP-Happiness data (as you did).

## Week 3: Data Preprocessing
- **Book Chapter**: Chapter 2 (Sections on Data Cleaning, Feature Engineering)
- **Topics**: Handling missing data, outliers, categorical data. Feature scaling (standardization, normalization). Column transformers.
- **Videos**:
  - Video 24: Feature Scaling - Standardization (32:38)
  - Video 25: Feature Scaling - Normalization (23:31)
  - Video 26: Encoding Categorical Data (19:53)
  - Video 27: One Hot Encoding (30:12)
  - Video 36: Handling missing data | Numerical Data (31:21)
  - Video 37: Handling Missing Categorical Data (13:34)
- **Goals**: Master data cleaning and preprocessing. Apply to real datasets.
- **Practice**: Clean your GDP-Happiness dataset (e.g., handle NaNs, scale GDP).

## Week 4: Classification Basics
- **Book Chapter**: Chapter 3: Classification
- **Topics**: Classification tasks (e.g., MNIST). Performance metrics (accuracy, confusion matrix, precision, recall, F1, ROC-AUC).
- **Videos**:
  - Video 75: Accuracy and Confusion Matrix (34:08)
  - Video 76: Precision, Recall and F1 Score (42:42)
  - Video 134: ROC Curve in Machine Learning (1:11:15)
- **Goals**: Understand classification and evaluation metrics. Train a simple classifier.
- **Practice**: Run MNIST classification code from Chapter 3. Try classifying a Kaggle dataset.

## Week 5: Linear Regression
- **Book Chapter**: Chapter 4: Training Models (Sections on Linear Regression)
- **Topics**: Linear regression. Cost functions (MSE). Regression metrics (MAE, RMSE, R²). Gradient descent basics.
- **Videos**:
  - Video 50: Simple Linear Regression (33:36)
  - Video 51: Simple Linear Regression | Mathematical Formulation (53:31)
  - Video 52: Regression Metrics (43:56)
- **Goals**: Master linear regression and metrics. Visualize results (as you did with GDP-Happiness).
- **Practice**: Implement regression on your GDP-Happiness data. Plot regression line (Week 3 code).

## Week 6: Optimization and Regularization
- **Book Chapter**: Chapter 4 (Sections on Gradient Descent, Regularized Models)
- **Topics**: Gradient descent (batch, stochastic, mini-batch). Regularized models (Ridge, Lasso, ElasticNet). Bias/variance trade-off.
- **Videos**:
  - Video 56: Gradient Descent From Scratch (1:57:56)
  - Video 57: Batch Gradient Descent (1:04:49)
  - Video 58: Stochastic Gradient Descent (49:35)
  - Video 59: Mini-Batch Gradient Descent (22:10)
  - Video 61: Bias Variance Trade-off (8:05)
  - Video 62: Ridge Regression Part 1 (19:58)
  - Video 66: Lasso Regression (28:37)
- **Goals**: Learn optimization techniques and prevent overfitting.
- **Practice**: Apply Ridge regression to your dataset. Compare with linear regression.

## Week 7: Logistic Regression
- **Book Chapter**: Chapter 4 (Sections on Logistic Regression); Chapter 3 (Classification Metrics)
- **Topics**: Logistic regression. Sigmoid function. Binary cross-entropy loss. Softmax for multiclass.
- **Videos**:
  - Video 69: Logistic Regression Part 1 | Perceptron Trick (47:06)
  - Video 71: Logistic Regression Part 3 | Sigmoid Function (40:44)
  - Video 77: Softmax Regression (38:21)
- **Goals**: Understand logistic regression for classification tasks.
- **Practice**: Train logistic regression on a binary classification dataset (e.g., Kaggle’s Titanic).

## Week 8: Support Vector Machines
- **Book Chapter**: Chapter 5: Support Vector Machines
- **Topics**: SVMs (hard/soft margin, kernel trick). Linear vs. non-linear classification.
- **Videos**:
  - Video 113: Support Vector Machines | Geometric Intuition (11:46)
  - Video 114: Mathematics of SVM | Hard margin SVM (34:54)
  - Video 116: Kernel Trick in SVM | Code Example (14:04)
- **Goals**: Master SVMs for robust classification.
- **Practice**: Run SVM on MNIST or a Kaggle dataset.

## Week 9: Decision Trees and Ensemble Learning
- **Book Chapters**: Chapter 6: Decision Trees; Chapter 7: Ensemble Learning and Random Forests
- **Topics**: Decision trees (entropy, Gini impurity). Ensemble methods (bagging, random forests, boosting).
- **Videos**:
  - Video 80: Decision Trees Geometric Intuition (58:29)
  - Video 91: Introduction to Random Forest (33:55)
  - Video 102: Bagging Vs Boosting (6:17)
- **Goals**: Learn tree-based models and ensemble techniques.
- **Practice**: Build a random forest model on a Kaggle dataset.

## Week 10: Dimensionality Reduction
- **Book Chapter**: Chapter 8: Dimensionality Reduction
- **Topics**: PCA, t-SNE. Curse of dimensionality. Visualization techniques.
- **Videos**:
  - Video 46: Curse of Dimensionality (15:25)
  - Video 47: Principle Component Analysis (PCA) | Part 1 (33:54)
  - Video 49: Principle Component Analysis(PCA) | Part 3 (43:26)
- **Goals**: Reduce data dimensions and visualize results (e.g., your Walmart scenario).
- **Practice**: Apply PCA to your GDP-Happiness data and visualize.

## Week 11: Unsupervised Learning
- **Book Chapter**: Chapter 9: Unsupervised Learning Techniques
- **Topics**: Clustering (K-Means, Hierarchical, DBSCAN). Anomaly detection.
- **Videos**:
  - Video 103: K-Means Clustering Algorithm (23:58)
  - Video 110: Agglomerative Hierarchical Clustering (37:23)
  - Video 131: DBSCAN Clustering Algorithms (34:16)
- **Goals**: Understand clustering and its applications (e.g., customer segmentation).
- **Practice**: Cluster your GDP-Happiness data (e.g., group countries by GDP and happiness).

## Week 12: Advanced Topics and Review
- **Book Chapters**: Chapter 2 (Hyperparameter Tuning); Chapter 3 (ROC-AUC); Chapter 9 (Anomaly Detection)
- **Topics**: Hyperparameter tuning (GridSearchCV). Handling imbalanced data. ML challenges (e.g., sampling bias, noise).
- **Videos**:
  - Video 132: Imbalanced Data in Machine Learning (57:17)
  - Video 133: Hyperparameter Tuning using Optuna (59:23)
- **Goals**: Optimize models and address real-world issues. Review roadmap progress.
- **Practice**: Tune a model on your dataset. Try a Kaggle competition.

## Notes
- **Practice**: Follow book code examples in Jupyter Notebook. Apply concepts to your GDP-Happiness project or Kaggle datasets.
- **Tools**: Set up Anaconda (Video 12) in Week 2. Use Google Colab if needed.
- **Approach**: Read chapter first, watch videos for reinforcement, then code.
- **Supplementary Videos**: Watch Video 8 (Applications, 29:02) and Video 10 (Job Roles, 26:23) anytime for motivation.
- **Next Steps**: Explore deep learning (Part II of book) or start a research project with a professor.