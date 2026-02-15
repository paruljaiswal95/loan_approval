a. Problem Statement – Loan Approval Prediction
The task focuses on determining whether a loan application will be accepted or rejected. Using applicant details provided by a housing finance company, the objective is to build a predictive model that classifies applications into approved or not approved.

b. Dataset Overview
This dataset represents a binary classification problem. It contains demographic, financial, and credit-related attributes of loan applicants.

Numerical Features:

ApplicantIncome: Income of the primary applicant

CoapplicantIncome: Income contributed by a co-applicant

LoanAmount: Requested loan amount

Loan_Amount_Term: Duration of the loan in months

Categorical Features:

Gender

Marital Status

Dependents

Education

Self_Employed

Property_Area

Credit_History: Indicator of creditworthiness (1 = good, 0 = poor)

Target Variable

Loan_Status: Approved (Y) or Not Approved (N)

c. Model Evaluation
Several machine learning algorithms were applied, and their performance was assessed using accuracy, AUC, precision, recall, F1-score, and MCC.

Model                 Accuracy    AUC       Precision   Recall     F1 Score   MCC
-------------------------------------------------------------------------------
Logistic Regression   0.837398   0.766873   0.809524    1.000000   0.894737   0.619240
Decision Tree         0.674797   0.715789   0.800000    0.705882   0.750000   0.294723
KNN                   0.821138   0.696749   0.800000    0.988235   0.884211   0.569458
Naive Bayes           0.837398   0.763467   0.815534    0.988235   0.893617   0.611360
Random Forest         0.813008   0.771053   0.810000    0.952941   0.875676   0.536759
XGBoost               0.829268   0.777399   0.813725    0.976471   0.887701   0.585097

d. Observations
Model                Key Insights
-------------------------------------------------------------------------------
Logistic Regression  Excellent recall (1.0) with strong accuracy (0.84) and F1 (0.89).
                     Very reliable at identifying approved loans, though precision
                     shows some false positives.

Decision Tree        Weakest performer overall. Accuracy (0.67) and MCC (0.29) are low.
                     Recall is modest (0.71), suggesting instability and possible
                     overfitting compared to ensemble methods.

kNN                  High accuracy (0.82) and recall (0.99). Balanced F1 (0.88) and
                     MCC (0.57). Performs strongly, though AUC is lower, meaning
                     weaker class separation.

Naive Bayes          Matches Logistic Regression closely. Accuracy (0.84), recall (0.99),
                     and F1 (0.89) are excellent. MCC (0.61) indicates solid correlation.
                     A strong baseline model.

Random Forest        Stable ensemble model. Accuracy (0.81) and recall (0.95) are strong.
                     F1 (0.88) is competitive, but MCC (0.54) is slightly weaker than
                     top performers.

XGBoost              Best overall balance. Accuracy (0.83), recall (0.98), and F1 (0.89)
                     are all strong. Highest AUC (0.78) shows superior class separation
                     and generalization ability.

e. Conclusion
Among all tested models, XGBoost and kNN stand out as the most effective, consistently achieving high accuracy, recall, and F1-scores. XGBoost edges ahead with superior AUC, indicating better ability to distinguish between approved and rejected applications. Logistic Regression and Naive Bayes serve as reliable baselines, while Decision Tree lags due to stability issues. Ensemble methods (Random Forest, XGBoost) generally provide stronger generalization compared to single models.