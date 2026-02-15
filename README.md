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

Model               Accuracy   AUC     Precision   Recall   F1 Score   MCC
-------------------------------------------------------------------------------
Logistic Regression   0.790    0.733     0.800      0.930     0.860     0.471
Decision Tree         0.758    0.729     0.769      0.930     0.842     0.374
KNN                   0.806    0.733     0.804      0.953     0.872     0.516
Naive Bayes           0.774    0.743     0.796      0.907     0.848     0.431
Random Forest         0.758    0.747     0.792      0.884     0.835     0.394
XGBoost               0.806    0.781     0.804      0.953     0.872     0.516

d. Observations
Model                Key Insights
-------------------------------------------------------------------------------
Logistic Regression  Strong recall (0.93) and balanced precision (0.80).
                     Produces a solid F1-score (0.86). AUC is moderate,
                     showing limited separation compared to XGBoost.

Decision Tree        Recall is high, but precision and MCC are weaker.
                     Model may be prone to overfitting and less consistent
                     than ensemble methods.

kNN                  Delivers high accuracy (0.81) and recall (0.95).
                     Strong F1-score (0.87) and MCC (0.52) suggest balanced
                     performance. One of the top-performing models.

Naive Bayes          Good recall (0.91) with moderate precision and AUC.
                     Performs reasonably well but slightly behind kNN
                     and XGBoost.

Random Forest        Provides stable results with decent AUC (0.75),
                     but accuracy and F1 are lower than kNN and XGBoost.

XGBoost              Best overall performer. Highest AUC (0.78), recall (0.95),
                     F1-score (0.87), and MCC (0.52). Shows strong generalization
                     and class separation ability.

e. Conclusion
Among all tested models, XGBoost and kNN stand out as the most effective, consistently achieving high accuracy, recall, and F1-scores. XGBoost edges ahead with superior AUC, indicating better ability to distinguish between approved and rejected applications. Logistic Regression and Naive Bayes serve as reliable baselines, while Decision Tree lags due to stability issues. Ensemble methods (Random Forest, XGBoost) generally provide stronger generalization compared to single models.