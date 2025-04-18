# 🚦 Predicting Arrests from Traffic Stops – A Data Science Project

## 📌 Project Overview
This project explores the police stop data (`police.csv`) to build a classification model that predicts whether a driver will be arrested during a traffic stop. The dataset includes various features like gender, race, violation type, and stop duration. Our main objective was to clean and preprocess this real-world dataset and apply machine learning models to understand the patterns and challenges in prediction.

---

## 🧹 Preprocessing & EDA
- Dropped columns and rows with missing values.
- Applied one-hot encoding for non-ordinal categorical features and ordinal encoding for others.
- Conducted exploratory data analysis (EDA) with visualizations to understand feature distributions and correlations.

---

## 🧠 Model Building – Logistic Regression
Initially, we chose **Logistic Regression** to tackle the binary classification task: **Will a driver be arrested (True/False)?**

### 🔄 Imbalance Handling
The dataset was significantly imbalanced (many more "not arrested" cases), so we tried:
- `class_weight='balanced'` in Logistic Regression.
- SMOTE (Synthetic Minority Oversampling Technique).

---

## 🔍 Hyperparameter Tuning
Using `GridSearchCV`, we found the best parameters:
```python
{'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
```

---

## 📊 Evaluation Metrics (Best Logistic Regression Model)
### Confusion Matrix:
```
[[15357  2382]
 [  176   528]]
```

### Classification Report:
```
              precision    recall  f1-score   support

       False       0.99      0.87      0.92     17739
        True       0.18      0.75      0.29       704

    accuracy                           0.86     18443
   macro avg       0.59      0.81      0.61     18443
weighted avg       0.96      0.86      0.90     18443
```

### Accuracy Score:
```python
0.8612
```

Despite the decent accuracy, the low precision and recall for "True" (arrested) show how imbalanced data can affect model performance.

---

## 💡 Lessons Learned
- Cleaning real-world data can be messy and time-consuming.
- Imbalanced datasets require special handling techniques like SMOTE and class weighting.
- Logistic regression, though simple, provides a strong foundation for classification.
- Evaluation beyond accuracy is crucial – especially with imbalanced classes.

---

## 📁 Project Structure
```
📦police-stop-analysis
 ┣ 📄 01_preprocessing_and_eda.ipynb
 ┣ 📄 02_logistic_model.ipynb
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## 🤝 Acknowledgment
Thanks to the open dataset (`police.csv`) and the many helpful tutorials and guides that made learning possible.

---

## 📬 Contact
Feel free to connect or reach out on [LinkedIn](http://linkedin.com/in/anthonykennetho)) or GitHub: [@anthonykennetho](https://github.com/anthonykennetho)

---

## 🧠 Next Steps
- Explore more classification algorithms (e.g., XGBoost, SVM).
- Try ensemble methods and advanced feature engineering.
- Build a dashboard to visualize predictions and insights interactively.
