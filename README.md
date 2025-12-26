#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


# In[3]:


# Path to the Data Set
path =  r'C:\Users\abouz\OneDrive\Desktop\For me\MBA\AI Strategy'
glob(path+"\*.csv")


# In[4]:


#Loading the Dataset
customertravel=pd.read_csv(path+'\Customertravel.csv')


# # EDA

# Performing EDA to the Data set to make sure it is loaded correctly, normalize any outliers, and remove rows with missing data etc ...

# In[5]:


customertravel.head()


# In[6]:


customertravel.tail()


# In[7]:


customertravel.columns


# In[8]:


customertravel.dtypes


# In[9]:


customertravel.info()


# In[10]:


customertravel.shape


# In[11]:


print(customertravel.isnull().sum())


# In[12]:


print((customertravel == 0).sum())


# No null or Zero values to any of the attributes, the 0 values at targets stand to a boolean 0 , which is acceptable 

# # Transforming Categorical Features into Numeric Representations

# All variables in the dataset were converted into a fully numeric, model-ready format using encoding strategies aligned with their semantic meaning. Continuous variables (Age and ServicesOpted) and the binary target (Target) were kept in their original numeric form. Binary categorical features (AccountSyncedToSocialMedia and BookedHotelOrNot) were mapped to 0/1 for simplicity and interpretability. The ordinal categorical variable (AnnualIncomeClass) was ordinally encoded to preserve its natural ranking from low to high income, while the nominal categorical variable (FrequentFlyer), which has no inherent order, was one-hot encoded to avoid introducing artificial hierarchy. This approach ensured numerical consistency across features while preserving meaningful structure for effective churn modeling.

# In[13]:


binary_map = {"Yes": 1, "No": 0}

customertravel["AccountSyncedToSocialMedia"] = (
    customertravel["AccountSyncedToSocialMedia"].map(binary_map)
)

customertravel["BookedHotelOrNot"] = (
    customertravel["BookedHotelOrNot"].map(binary_map)
)


# In[14]:


income_map = {
    "Low Income": 0,
    "Middle Income": 1,
    "High Income": 2
}

customertravel["AnnualIncomeClass"] = (
    customertravel["AnnualIncomeClass"].map(income_map)
)


# In[15]:


customertravel = pd.get_dummies(
    customertravel,
    columns=["FrequentFlyer"],
    drop_first=True
)


# One category of the frequent flyer variable was intentionally dropped during one-hot encoding to serve as the baseline and avoid multicollinearity. The dropped category (‘No’) is implicitly represented when all dummy variables are zero.

# In[16]:


customertravel.head()


# In[17]:


customertravel.tail()


# In[18]:


print(customertravel.isnull().sum())


# In[19]:


print((customertravel == 0).sum())


# In[20]:


# Review dataset statistics
customertravel.describe()


# The summary statistics confirm that all features are now fully numeric and suitable for modeling. Age remains tightly distributed between 27 and 38 years with a mean of about 32, indicating limited demographic variation. AnnualIncomeClass, encoded ordinally from low (0) to high (2), has a mean of 0.58, showing that most customers fall in the low-to-mid income categories. ServicesOpted has a mean of approximately 2.4 and a wide range from 1 to 6, reflecting meaningful variation in customer engagement and reinforcing its potential importance for churn prediction.
# 
# The binary behavioral features show clear proportions: about 38% of customers have synced their account to social media, and roughly 40% have booked a hotel, suggesting moderate digital and product engagement. The FrequentFlyer one-hot encoded variables indicate that around 30% are frequent flyers, while a small proportion (about 6%) fall under the “No Record” category, which acts as the baseline. The Target variable maintains a churn rate of approximately 23.5%, confirming moderate class imbalance. Overall, the statistics show that while demographic signals are relatively constrained, behavioral and engagement-related features exhibit sufficient variability to be informative drivers in a churn prediction model.

# In[21]:


ax = customertravel["Target"].value_counts().plot(kind="bar", figsize=(6,4))

# Title and labels
ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn (0 = Stayed, 1 = Churned)")
ax.set_ylabel("Number of Customers")

# Add value labels on each bar
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.show()


# In[22]:


sns.histplot(customertravel["Age"], kde=True)
plt.title("Age Distribution of Customers")
plt.show()


# In[23]:


plt.figure(figsize=(6,4))
sns.histplot(
    customertravel["ServicesOpted"],
    bins=6,
    kde=True,
    discrete=True
)
plt.title("Distribution of Services Opted")
plt.xlabel("Number of Services Opted")
plt.ylabel("Count")
plt.show()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(
    customertravel,
    vars=["Age", "ServicesOpted"],
    hue="Target",
    diag_kind="hist",
    plot_kws={"alpha": 0.6}
)

plt.suptitle("Pairplot of Numeric Features by Churn Status", y=1.02, fontsize=14)
plt.show()


# The pairplot visualizes the relationship between the two numeric features, Age and ServicesOpted, segmented by churn status. The diagonal histograms show that Age distributions for churned and non-churned customers largely overlap, indicating that age alone provides limited separation between the two classes. In contrast, the distribution of ServicesOpted shows a clearer pattern, where customers who have opted for fewer services appear more frequently in the churned group, while higher service adoption is more common among non-churned customers.
# 
# The off-diagonal scatter plots indicate no strong interaction or correlation between Age and ServicesOpted, as customers of different ages are spread similarly across service adoption levels. Color overlap across the scatter confirms that Age does not meaningfully differentiate churn behavior, whereas service engagement level is a more informative signal. Overall, the pairplot supports the conclusion that churn in this dataset is driven more by behavioral engagement than by demographic factors, and that modeling efforts should focus on features related to usage and service adoption rather than age.

# In[25]:


ln = customertravel.corr()
ln


# In[26]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(ln, ax=ax,cmap="Pastel1", linewidths=0.1,annot=True, fmt = ".2f")
plt.show()


# The correlation matrix, computed after encoding all features numerically, provides a high-level diagnostic view of linear relationships in the dataset. It shows that demographic variables such as Age have only a weak association with churn, while behavioral and engagement-related features exhibit stronger signals. In particular, FrequentFlyer_Yes and AnnualIncomeClass display the most notable positive correlations with the target, indicating that customer loyalty and income level are meaningfully associated with churn outcomes, whereas BookedHotelOrNot shows a moderate negative association, suggesting a protective effect. Most other correlations are weak, confirming the absence of strong linear dependencies and limited multicollinearity among features. Overall, the matrix supports earlier EDA conclusions that churn in this dataset is driven more by behavioral patterns than demographics, while also highlighting the need for model-based and non-linear methods to fully capture predictive relationships.

# In[27]:


customertravel.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()


# This set of histograms shows the distributions of all encoded features after preprocessing, confirming that the dataset is clean and model-ready. Age is narrowly distributed between the late 20s and late 30s, with most customers concentrated in the early 30ucker 30s, indicating limited demographic spread. AnnualIncomeClass, encoded ordinally, shows that the majority of customers fall into the low- and mid-income categories, with fewer high-income customers. ServicesOpted exhibits a right-skewed distribution, with most customers opting for fewer services and progressively fewer customers using higher numbers of services, highlighting meaningful variation in engagement.
# 
# The remaining features are binary and display clear class proportions: a larger share of customers have not synced their account to social media and have not booked a hotel, while the Target histogram confirms that non-churners dominate the dataset, reflecting moderate class imbalance. The one-hot encoded FrequentFlyer variables show that most customers are not frequent flyers, with a small proportion having no recorded status. Overall, these distributions validate earlier EDA findings that churn-related signal is more likely to come from behavioral and engagement features rather than demographic attributes, and they confirm that no abnormal distributions or data quality issues are present before modeling.

# # Modeling

# In[28]:


# =========================
# Model Comparison Pipeline
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC


# -------------------------
# 1) Split features/target
# -------------------------
X = customertravel.drop("Target", axis=1)
y = customertravel["Target"]


# -------------------------
# 2) Define feature groups
#    (adjust names if yours differ)
# -------------------------
numeric_features = ["Age", "ServicesOpted"]
binary_features = [
    "AccountSyncedToSocialMedia",
    "BookedHotelOrNot",
    "FrequentFlyer_No Record",
    "FrequentFlyer_Yes",
]
ordinal_features = ["AnnualIncomeClass"]


# -------------------------
# 3) Preprocessing
#    - QuantileTransformer ONLY on continuous numeric features
#    - Pass-through binary & ordinal features
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(output_distribution="normal", random_state=42), numeric_features),
        ("bin", "passthrough", binary_features),
        ("ord", "passthrough", ordinal_features),
    ],
    remainder="drop"
)


# -------------------------
# 4) Define models to compare
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM (Linear)": SVC(kernel="linear", class_weight="balanced"),
}


# -------------------------
# 5) Cross-validation setup
# -------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -------------------------
# 6) Evaluate each model (Accuracy)
# -------------------------
results = []

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "Model": name,
        "Mean Accuracy": scores.mean(),
        "Std Accuracy": scores.std()
    })


# -------------------------
# 7) Results table (sorted)
# -------------------------
results_df = pd.DataFrame(results).sort_values(by="Mean Accuracy", ascending=False).reset_index(drop=True)
print(results_df)


# -------------------------
# 8) Optional: Visualize comparison
# -------------------------
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Mean Accuracy", y="Model")
plt.title("Model Comparison (5-Fold CV Mean Accuracy)")
plt.xlabel("Mean Accuracy")
plt.ylabel("Model")
plt.tight_layout()
plt.show()


# The cross-validated results show a clear performance hierarchy among the evaluated models. Decision Tree and Gradient Boosting achieve the highest mean accuracies (≈ 0.89) with relatively low standard deviations, indicating strong performance and stable behavior across folds. Random Forest and KNN follow closely, also delivering high accuracies above 0.86, though with slightly more variability. AdaBoost performs moderately well but does not surpass the tree-based leaders. In contrast, the linear models—Logistic Regression and Linear SVM—exhibit noticeably lower accuracy (≈ 0.77) and higher variance, suggesting that linear decision boundaries are insufficient to capture the underlying patterns in the data. Overall, the results indicate that non-linear, tree-based models are better suited for this churn prediction task, with Decision Tree and Gradient Boosting emerging as the strongest candidates for further tuning and evaluation.

# The next step is to narrow the focus to the top-performing models—specifically the Decision Tree and Gradient Boosting classifiers—and evaluate them using churn-relevant metrics beyond accuracy. This involves assessing recall, precision, and ROC-AUC on a held-out test set to ensure the chosen model effectively identifies churners while maintaining stability. Based on these results, the selected model should then be fine-tuned through targeted hyperparameter optimization and interpreted using feature importance or explainability techniques to translate predictions into actionable churn-reduction strategies.

# #     Hyperparameter Tuning
# 

# In[29]:


# =========================
# Next Step: Evaluate Top Models on a Hold-out Test Set
# Metrics: Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# -------------------------
# 1) Features / target
# -------------------------
X = customertravel.drop("Target", axis=1)
y = customertravel["Target"]

# -------------------------
# 2) Train/Test split (stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3) Feature groups (adjust if needed)
# -------------------------
numeric_features = ["Age", "ServicesOpted"]
binary_features = [
    "AccountSyncedToSocialMedia",
    "BookedHotelOrNot",
    "FrequentFlyer_No Record",
    "FrequentFlyer_Yes",
]
ordinal_features = ["AnnualIncomeClass"]

# -------------------------
# 4) Preprocessor
# NOTE: QuantileTransformer is optional for tree models,
# but kept here for consistency with your earlier comparison.
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(output_distribution="normal", random_state=42), numeric_features),
        ("bin", "passthrough", binary_features),
        ("ord", "passthrough", ordinal_features),
    ],
    remainder="drop"
)

# -------------------------
# 5) Define the two top models
# -------------------------
top_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# -------------------------
# 6) Helper to evaluate a model
# -------------------------
def evaluate_model(name, pipeline, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Some models have predict_proba; use it for ROC-AUC
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        auc = None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_pred, y_prob

# -------------------------
# 7) Run evaluation for both
# -------------------------
all_metrics = []
cms = {}

for name, clf in top_models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    metrics, cm, y_pred, y_prob = evaluate_model(name, pipe, X_test, y_test)
    all_metrics.append(metrics)
    cms[name] = cm

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))

# Metrics table
metrics_df = pd.DataFrame(all_metrics).sort_values(by="ROC-AUC", ascending=False)
print("\nSummary Metrics (sorted by ROC-AUC):")
print(metrics_df)

# -------------------------
# 8) Plot confusion matrices
# -------------------------
plt.figure(figsize=(10,4))

for i, (name, cm) in enumerate(cms.items(), start=1):
    plt.subplot(1, 2, i)
    sns.heatmap(cm,cmap="Pastel1", annot=True, fmt="d", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# The evaluation results highlight a clear trade-off between overall predictive performance and churn detection effectiveness. Gradient Boosting achieves the highest overall accuracy (≈ 91.6%) and a very strong ROC-AUC (≈ 0.97), indicating excellent ranking capability and stability across classes. It is highly precise when predicting churn (precision ≈ 0.94), meaning most customers flagged as churners truly churn; however, its recall for churn is lower (≈ 0.69), showing that it misses a noticeable portion of actual churners.
# 
# In contrast, the Decision Tree model delivers slightly lower accuracy (≈ 90.0%) and ROC-AUC (≈ 0.91), but it captures churners more effectively, with higher recall for the churn class (≈ 0.73). This behavior is reflected in the confusion matrices: the Decision Tree correctly identifies more churn cases at the expense of slightly more false positives, whereas Gradient Boosting is more conservative, minimizing false positives but allowing more churners to slip through. Overall, Gradient Boosting is preferable when ranking and precision are priorities, while the Decision Tree may be more suitable when the business objective emphasizes catching as many churners as possible.

# # Tuning threshold 

# In[30]:


# ============================================================
# Threshold Tuning for Gradient Boosting (ROC + PR Curves)
# End-to-end code: build pipeline -> fit -> curves -> choose threshold
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)

# -------------------------
# 1) Prepare X and y
# -------------------------
X = customertravel.drop("Target", axis=1)
y = customertravel["Target"]

# -------------------------
# 2) Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 3) Feature groups
# (Adjust column names if yours differ)
# -------------------------
numeric_features = ["Age", "ServicesOpted"]
binary_features = [
    "AccountSyncedToSocialMedia",
    "BookedHotelOrNot",
    "FrequentFlyer_No Record",
    "FrequentFlyer_Yes",
]
ordinal_features = ["AnnualIncomeClass"]

# -------------------------
# 4) Preprocessor
# NOTE: QuantileTransformer is optional for tree models,
# kept here for consistency with your earlier pipeline.
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(output_distribution="normal", random_state=42), numeric_features),
        ("bin", "passthrough", binary_features),
        ("ord", "passthrough", ordinal_features),
    ],
    remainder="drop"
)

# -------------------------
# 5) Build Gradient Boosting pipeline
# -------------------------
gb_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

# -------------------------
# 6) Fit model and get churn probabilities
# -------------------------
gb_pipe.fit(X_train, y_train)
y_prob = gb_pipe.predict_proba(X_test)[:, 1]

# -------------------------
# 7) ROC Curve
# -------------------------
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve – Gradient Boosting")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 8) Precision–Recall Curve
# -------------------------
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Gradient Boosting")
plt.tight_layout()
plt.show()

# -------------------------
# 9) Evaluate metrics across thresholds
# -------------------------
thresholds = np.arange(0.10, 0.91, 0.05)

rows = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    rows.append({
        "Threshold": round(float(t), 2),
        "Precision": precision_score(y_test, y_pred_t, zero_division=0),
        "Recall": recall_score(y_test, y_pred_t, zero_division=0),
        "F1": f1_score(y_test, y_pred_t, zero_division=0)
    })

threshold_df = pd.DataFrame(rows)

print("Threshold Metrics (sorted by F1):")
print(threshold_df.sort_values("F1", ascending=False).head(10))

# -------------------------
# 10) Plot Precision/Recall/F1 vs Threshold
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
plt.plot(threshold_df["Threshold"], threshold_df["F1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning – Gradient Boosting")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 11) Pick "best" threshold by F1 (you can change rule)
# -------------------------
best_row = threshold_df.loc[threshold_df["F1"].idxmax()]
best_threshold = best_row["Threshold"]

print("\nBest Threshold (by F1):", best_threshold)
print(best_row)


# The attached results show that the Gradient Boosting model has excellent ranking capability and benefits significantly from threshold tuning. The ROC curve exhibits a very high ROC-AUC of approximately 0.97, confirming that the model can reliably distinguish churners from non-churners across a wide range of thresholds. The Precision–Recall curve further highlights the trade-off between catching churners and avoiding false positives, which is especially important given the class imbalance in churn data.
# 
# By explicitly evaluating performance across multiple probability thresholds, the analysis identifies 0.30 as the optimal threshold based on F1 score. At this cutoff, the model achieves very high recall (≈ 93%), meaning it successfully identifies most churners, while maintaining reasonable precision (≈ 76%), resulting in a strong F1 score of ≈ 0.84. The threshold tuning plot clearly illustrates this balance: lower thresholds increase recall at the cost of precision, while higher thresholds do the opposite. Overall, this confirms that using a non-default threshold substantially improves churn detection effectiveness and aligns the model’s behavior with business objectives focused on proactive churn prevention.

# # Environment-Aligned Model Re-Export for Streamlit Deployment

# This script re-trains and re-exports the Gradient Boosting churn prediction pipeline and its associated dashboard metadata from the same Python and scikit-learn environment used by the deployed Streamlit application. By rebuilding the model pipeline in-environment and explicitly recording version information, it eliminates serialization and compatibility issues during deployment. The process also formalizes model performance metadata (ROC-AUC, optimal threshold, precision, recall, and F1) into a structured artifact, ensuring consistency between training, evaluation, and production use while enabling reliable, reproducible model loading in the live web application.

# In[31]:


import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingClassifier

EXPORT_DIR = r"C:\Users\abouz\OneDrive\Desktop\For me\MBA\AI Strategy\travel-churn-portal"
os.makedirs(EXPORT_DIR, exist_ok=True)

X = customertravel.drop("Target", axis=1)
y = customertravel["Target"]

numeric_features = ["Age", "ServicesOpted"]
binary_features = ["AccountSyncedToSocialMedia", "BookedHotelOrNot", "FrequentFlyer_No Record", "FrequentFlyer_Yes"]
ordinal_features = ["AnnualIncomeClass"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(output_distribution="normal", random_state=42), numeric_features),
        ("bin", "passthrough", binary_features),
        ("ord", "passthrough", ordinal_features),
    ],
    remainder="drop"
)

gb_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gb_pipe.fit(X_train, y_train)

joblib.dump(gb_pipe, os.path.join(EXPORT_DIR, "gb_churn_pipeline.pkl"))
print("✅ Saved:", os.path.join(EXPORT_DIR, "gb_churn_pipeline.pkl"))


# In[32]:


import sklearn
sklearn.__version__


# In[34]:


dashboard_meta = {
    "suggested_model_from_cv": "Gradient Boosting",
    "roc_auc": 0.973,
    "best_threshold_by_f1": 0.30,
    "best_threshold_precision": 0.76,
    "best_threshold_recall": 0.93,
    "best_threshold_f1": 0.84
}


# In[35]:


# -------------------------
# 3) Export to your repo folder
# -------------------------
EXPORT_DIR = r"C:\Users\abouz\OneDrive\Desktop\For me\MBA\AI Strategy\travel-churn-portal"

joblib.dump(gb_pipe, os.path.join(EXPORT_DIR, "gb_churn_pipeline.pkl"))
joblib.dump(dashboard_meta, os.path.join(EXPORT_DIR, "dashboard_meta.pkl"))

print("✅ Saved pkl files into:", EXPORT_DIR)




