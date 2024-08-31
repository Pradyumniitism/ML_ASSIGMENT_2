
# Machine Learning Assignment 2

## Name: Pradyumn Sharma  
## Submitted to: Prof. Sandeep Mondal  
### Roll No: 23MT0263  
### Roll No: 23MT0263@IITISM.AC.IN

## Overview

This assignment involves implementing various classification algorithms to analyze a dataset and compare their performance. The dataset provided is utilized to perform the following classification techniques:

1. Logistic Regression
2. K-Nearest Neighbors (kNN)
3. Decision Tree
4. Naive Bayes
5. Support Vector Machine (SVM)

The goal is to evaluate these models using metrics such as accuracy, confusion matrix, and AUC (Area Under Curve), and to compare their performance.

## Dataset

The dataset used is `diabetes_dataset.xlsx`, which contains information relevant for classification.

## Data Exploration

1. **Loading Data**:  
   ```python
   import pandas as pd
   
   df = pd.read_excel(r"C:\Users\Pradyumn Sharma\Desktop\MACHINE LEARNING ASSIGNMENT\ASSIGNMENT 1\assi1dia\diabetes_dataset.xlsx")
   ```

2. **Exploring Data**:  
   - **Shape and Description**:
     ```python
     df.shape
     df.describe()
     df.info()
     ```
   - **Target Variable**:
     ```python
     df.iloc[:, 7].value_counts()
     ```
   - **Correlation and Visualization**:
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt
     
     sns.pairplot(df)
     plt.show()
     
     grid = sns.PairGrid(df)
     grid.map_upper(sns.scatterplot)
     grid.map_lower(sns.kdeplot)
     grid.map_diag(sns.histplot)
     plt.show()
     ```

## Data Preprocessing

1. **Normalization**:
   ```python
   ndf = (df - df.min()) / (df.max() - df.min())
   X_n = ndf.drop('Outcome', axis=1)
   Y_n = ndf['Outcome']
   ```

2. **Splitting Data**:
   ```python
   from sklearn.model_selection import train_test_split
   
   X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_n, Y_n, test_size=0.2, random_state=10)
   ```

## Classification Techniques

1. **Logistic Regression**:
   ```python
   from sklearn.linear_model import LogisticRegression
   
   model = LogisticRegression(random_state=10)
   model.fit(X_Train, Y_Train)
   
   Y_Pred_Logistic = model.predict(X_Test)
   ```

2. **K-Nearest Neighbors (kNN)**:
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.model_selection import RandomizedSearchCV
   from scipy.stats import randint
   
   knn = KNeighborsClassifier()
   param_dist_knn = {"weights": ['uniform', 'distance'], "n_neighbors": randint(1, 10), "p": [2, 1]}
   clf_knn = RandomizedSearchCV(knn, param_dist_knn)
   clf_knn.fit(X_Train, Y_Train)
   
   Y_Pred_Knn = clf_knn.predict(X_Test)
   ```

3. **Decision Tree**:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.model_selection import RandomizedSearchCV
   
   d_tree = DecisionTreeClassifier()
   param_dist_tree = {
       'criterion': ['gini', 'entropy'],
       'class_weight': [{0: a_i, 1: b_i} for a_i in np.linspace(1, 1000, 10) for b_i in np.linspace(1, 1000, 10)]
   }
   clf_d_tree = RandomizedSearchCV(d_tree, param_dist_tree, n_iter=10, random_state=42)
   clf_d_tree.fit(X_Train, Y_Train)
   
   Y_Pre_d_Tree = clf_d_tree.predict(X_Test)
   ```

4. **Naive Bayes**:
   ```python
   from sklearn.naive_bayes import BernoulliNB
   
   clf_Naive_Bayes = BernoulliNB().fit(X_Train, Y_Train)
   Y_Pre_Naive_Bayes = clf_Naive_Bayes.predict(X_Test)
   ```

5. **Support Vector Machine (SVM)**:
   ```python
   from sklearn.svm import SVC
   from sklearn.model_selection import RandomizedSearchCV
   
   param_svm = {'C': np.linspace(0.001, 10, 10000)}
   clf_svm = RandomizedSearchCV(SVC(), param_svm)
   clf_svm.fit(X_Train, Y_Train)
   
   Y_Pred_SVM = clf_svm.predict(X_Test)
   ```

## Model Evaluation

1. **Accuracy and Confusion Matrix**:
   ```python
   from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
   
   Accuracy = pd.DataFrame([
       ['Logistic Regression', accuracy_score(Y_Test, Y_Pred_Logistic)],
       ['Naive Bayes', accuracy_score(Y_Test, Y_Pre_Naive_Bayes)],
       ['KNN', accuracy_score(Y_Test, Y_Pred_Knn)],
       ['Decision Tree', accuracy_score(Y_Test, Y_Pre_d_Tree)],
       ['SVM', accuracy_score(Y_Test, Y_Pred_SVM)]
   ], columns=['Classification Techniques', 'Accuracy'])
   
   Confusion_Matrix = pd.DataFrame([
       ['Logistic Regression', confusion_matrix(Y_Test, Y_Pred_Logistic).ravel()],
       ['Naive Bayes', confusion_matrix(Y_Test, Y_Pre_Naive_Bayes).ravel()],
       ['K-NN', confusion_matrix(Y_Test, Y_Pred_Knn).ravel()],
       ['Decision Tree', confusion_matrix(Y_Test, Y_Pre_d_Tree).ravel()],
       ['SVM', confusion_matrix(Y_Test, Y_Pred_SVM).ravel()]
   ], columns=['Classification Techniques', 'TN, FP, FN, TP'])
   
   # Plot Confusion Matrix
   ConfusionMatrixDisplay(confusion_matrix(Y_Test, Y_Pred_Logistic)).plot()
   ConfusionMatrixDisplay(confusion_matrix(Y_Test, Y_Pred_Knn)).plot()
   ConfusionMatrixDisplay(confusion_matrix(Y_Test, Y_Pred_SVM)).plot()
   ConfusionMatrixDisplay(confusion_matrix(Y_Test, Y_Pre_Naive_Bayes)).plot()
   ConfusionMatrixDisplay(confusion_matrix(Y_Test, Y_Pre_d_Tree)).plot()
   ```

2. **ROC Curve and AUC**:
   ```python
   from sklearn.metrics import roc_curve, auc
   
   fpr_logistic, tpr_logistic, _ = roc_curve(Y_Test, Y_Pred_Logistic)
   fpr_knn, tpr_knn, _ = roc_curve(Y_Test, Y_Pred_Knn)
   fpr_naivebayes, tpr_naivebayes, _ = roc_curve(Y_Test, Y_Pre_Naive_Bayes)
   fpr_decisiontree, tpr_decisiontree, _ = roc_curve(Y_Test, Y_Pre_d_Tree)
   fpr_svm, tpr_svm, _ = roc_curve(Y_Test, Y_Pred_SVM)
   
   plt.plot(fpr_logistic, tpr_logistic, label='Logistic Regression', color='grey')
   plt.plot(fpr_knn, tpr_knn, label='KNN', color='yellow')
   plt.plot(fpr_decisiontree, tpr_decisiontree, label='Decision Tree', color='green')
   plt.plot(fpr_svm, tpr_svm, label='SVM', color='black')
   plt.plot(fpr_naivebayes, tpr_naivebayes, label='Naive Bayes', color='pink')
   plt.legend()
   plt.title('ROC Curve')
   plt.show()
   
   auc_matrix = pd.DataFrame([
       ['Logistic Regression', auc(fpr_logistic, tpr_logistic)],
       ['Naive Bayes', auc(fpr_naivebayes, tpr_naivebayes)],
       ['KNN', auc(fpr_knn, tpr_knn)],
       ['Decision Tree', auc(fpr_decisiontree, tpr_decisiontree)],
       ['SVM', auc(fpr_svm, tpr_svm)]
   ], columns=['Classification Techniques', 'AUC Score'])
   ```

## Results and Interpretation

- **Logistic Regression**: AUC = 0.682, indicating moderate performance.
- **Naive Bayes**: AUC = 0.500, performing no better than random chance.
- **K-NN**: AUC = 0.679, similar performance to Logistic Regression.
- **Decision Tree**: AUC = 0.709, slightly better performance.
- **SVM**: AUC = 0.719, the highest AUC score, indicating the best performance among the models.

## Conclusion

The SVM model shows the best performance for this classification task, with the highest AUC score. Naive Bayes performs the least effectively, with an AUC score of 0.5. Decision Tree, Logistic Regression, and K-NN exhibit moderate performance.
