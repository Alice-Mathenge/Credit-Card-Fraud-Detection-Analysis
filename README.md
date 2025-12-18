# Credit-Card-Fraud-Detection-Analysis

![Fraud Detection Model Output](image-1.png)

## **Overview**
This project addresses the critical challenge of credit card fraud in the financial sector. As digital transactions surge, financial institutions face significant losses and operational inefficiencies from manual fraud detection. This repository contains a machine learning pipeline designed to automatically identify suspicious patterns and flag fraudulent transactions in real-time.

---

## **Business Understanding**
`Problem Statement:` How can a classification model accurately identify rare fraudulent cases while minimizing false alarms (false positives) that inconvenience legitimate customers?

`Primary Objective:` To develop a predictive model that distinguishes between fraudulent and legitimate transactions, reducing financial loss and improving security.


`Goal:` Minimize financial loss and operational costs while maintaining customer trust.

`Stakeholders:` Financial institutions, cardholders, and risk management teams.

---

## **Data Understanding**
`Source:` Credit Card Fraud Detection dataset (ULB/Kaggle)  
🔗 Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


`Observations:` 284,807 transactions.

`Features:` 31 features, including Time, Amount, and 28 PCA-transformed numerical variables (V1–V28).
> Note: Features V1–V28 are PCA-transformed variables provided by the dataset owners to protect sensitive customer information while preserving useful patterns for fraud detection.


`Target:` Class (0 for legitimate, 1 for fraudulent).

`Challenge:` The dataset is highly imbalanced, with fraudulent transactions accounting for less than 0.2% of all records.  
This motivated the use of specialized evaluation metrics (Recall, F1-score, PR-AUC) and resampling techniques (SMOTE).


---

## **Data Preparation**

`Data Cleaning:` Verified no missing values are present. Some duplicate transactions were identified, which were intentionally retained because removing them could eliminate valid repeated transactions or rare fraud cases. Keeping duplicates preserves the real-world transaction distribution and helps the model learn genuine fraud patterns more effectively.

`Scaling:` Used StandardScaler and MinMaxScaler to normalize the Time and Amount columns.

`Imbalance Handling:` Implemented SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data, ensuring the model learns the characteristics of the rare fraud class.

---

## **Tech Stack & Libraries**
- Data Handling: Pandas, NumPy.

- Visualization: Matplotlib, Seaborn.

- Preprocessing: Scikit-learn (StandardScaler, MinMaxScaler).

- Imbalance Handling: Imbalanced-learn (SMOTE).

- Modeling: XGBoost, Random Forest, LightGBM, Logistic Regression.

- Evaluation: Scikit-learn (Confusion Matrix, Precision-Recall Curves, ROC-AUC).

---

## **Methodology**
- `Exploratory Data Analysis (EDA):` Understanding patterns that differentiate fraud from normal transactions.

- `Preprocessing:` Scaling numerical features and handling extreme class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

- `Modeling:` Training multiple classifiers(Logistic Regression, LightGBM and XGBoost Classifier), with XGBoost emerging as the top-performing model.

- `Optimization:` Hyperparameter tuning via GridSearchCV.

- `Evaluation:` Prioritizing metrics like:    
    - `Confusion Matrix`, to visualize the trade-off between False positives and False Negatives.
    - `Precision-Recall AUC` to evaluate how well the model identifies the minority (fraud) class. 
    - `ROC-AUC Score` to measure the overall diagnostic ability of the classifiers.

---

## **Deployment & Insights**
`Final Model:` XGBoost was selected as the champion model due to its superior ability to detect fraud while maintaining efficiency.

`Reproducibility:` The pipeline includes all preprocessing steps (scaling and sampling) to ensure the model performs consistently in a production environment.

---

## **Model Performance Snapshot**
- Fraud Recall: ~87% (most fraudulent transactions detected)
- Fraud Precision: ~75% (reduced false alarms)
- ROC-AUC: High discrimination between fraud and normal transactions
- Threshold tuning applied to balance fraud detection and customer experience

---

## **Key Findings & Conclusion**
This project demonstrates that with careful handling of imbalanced data, feature preprocessing, and model validation, machine learning algorithms like XGBoost can effectively detect rare fraud events. The approach balances accuracy with business considerations, providing actionable insights for financial transaction monitoring.

---

## Contact
**Alice Wangui Mathenge**

**Phone:** 0727673400

**Email:** mathengealice709@gmail.com