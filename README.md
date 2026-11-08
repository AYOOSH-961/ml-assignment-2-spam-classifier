# ml-assignment-2-spam-classifier
# Spam Email Classification – ML Assignment 2

## a. Problem statement
Predict whether an email is spam (1) or not spam (0) from word, character and capital-letter frequency features, and compare five classification models on the same data.

## b. Dataset description
- Source: UCI Machine Learning Repository – Spambase (https://archive.ics.uci.edu/dataset/94/spambase)
- Instances: 4,601 emails (1,813 spam, 2,788 not spam)
- Features: 57 numeric features
  - 48 word frequencies (% of words in the email matching a word, e.g. "free", "money")
  - 6 character frequencies (e.g. "!", "$")
  - 3 capital-letter run lengths (average, longest, total)
- Target: is_spam (1 = spam, 0 = not spam)
- Split: 80% train (3,680 emails) and 20% test (921 emails), stratified by class. The test set is saved as test_data.csv.

## c. GitHub repository
https://github.com/AYOOSH-961/ml-assignment-2-spam-classifier

Live Streamlit app: https://2025ag50922ayoosh-spam-classifierg.streamlit.app

## d. Models used and comparison
All five models were trained on the same training split and evaluated on the same test split (921 emails).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9154 | 0.9029 | 0.8457 | 0.8734 | 0.7965 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on Accuracy (0.929), AUC (0.970), F1 (0.909) and MCC (0.852). Only 28 normal emails flagged and 37 spam missed. A linear boundary over word frequencies separates spam well. |
| Decision Tree | Accuracy 0.903 but lowest AUC (0.915) and low recall (0.846): missed 56 spam. A single depth-limited tree gives coarse probabilities, though it is the easiest model to interpret. |
| KNN | Mid-range (Accuracy 0.908, MCC 0.806): 41 false alarms, 44 missed spam. Relies on distance between scaled features, which becomes less informative with 57 features. |
| Naive Bayes | Lowest Accuracy (0.834), Precision (0.718) and MCC (0.694) but highest Recall (0.953): catches almost all spam (17 missed) but flags 136 normal emails. Its independence and normal-distribution assumptions do not fit skewed word frequencies. |
| Support Vector Machine | Nearly tied with Logistic Regression (Accuracy 0.927, MCC 0.847) with the highest Precision (0.928): fewest false alarms (25). The RBF kernel captures non-linear patterns but is slower and harder to explain. |
| Overall winner for this dataset | Logistic Regression: highest Accuracy, AUC, F1 and MCC, and simpler and faster than SVM. SVM is a close second if avoiding false alarms matters most. |

## Repository structure
- app.py – Streamlit app (upload test CSV, choose model, view metrics, confusion matrix and report)
- Model_Training.ipynb – training and evaluation notebook (run on BITS Virtual Lab)
- model/ – the five trained models (.pkl)
- test_data.csv – test data used for evaluation
- requirements.txt – Python libraries needed

## How to run locally
pip install -r requirements.txt
streamlit run app.py
