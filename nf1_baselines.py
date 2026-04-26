import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif

from tabpfn import TabPFNClassifier


# =========================================================
# Project settings
# =========================================================
FILE = "dataset-uci.xlsx"
SHEET_NAME = "Dataset"
TARGET = "Case Type"          # 1 = Familial, 0 = Sporadic
OUTPUT_DIR = "outputs"


# =========================================================
# Helper functions
# =========================================================
def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load the Excel dataset and remove the ID column if present.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Drop ID-like column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def save_df(df: pd.DataFrame, filename: str) -> None:
    """
    Save a dataframe to the outputs folder as CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


def get_group_difference_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Compare average feature values between sporadic (0) and familial (1) cases.
    This is useful for biological interpretation because it shows which symptoms
    differ the most between the two groups.
    """
    group_summary = df.groupby(target).mean(numeric_only=True).T

    # Rename columns for clarity
    group_summary.columns = ["sporadic_0", "familial_1"]

    # Absolute difference helps rank the strongest group differences
    group_summary["abs_diff"] = (
        group_summary["familial_1"] - group_summary["sporadic_0"]
    ).abs()

    group_summary = group_summary.sort_values("abs_diff", ascending=False)
    group_summary = group_summary.reset_index().rename(columns={"index": "feature"})

    return group_summary


def get_anova_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Rank features using ANOVA F-scores.
    The NF1 source paper used ANOVA to identify important symptom features
    before running classifiers.
    """
    X_imputed = X.copy()

    # Median imputation so ANOVA can run without missing-value errors
    for col in X_imputed.columns:
        X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())

    f_scores, p_values = f_classif(X_imputed, y)

    anova_df = pd.DataFrame({
        "feature": X_imputed.columns,
        "f_score": f_scores,
        "p_value": p_values
    }).sort_values("f_score", ascending=False)

    return anova_df.reset_index(drop=True)


def build_models():
    """
    Build baseline models and one recent model (TabPFN).
    """
    prep_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    prep_unscaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    models = {
        "LogReg_baseline": Pipeline([
            ("prep", prep_scaled),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "RandomForest_baseline": Pipeline([
            ("prep", prep_unscaled),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        "SVM_baseline": Pipeline([
            ("prep", prep_scaled),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced"))
        ]),
        "TabPFN_recent": Pipeline([
            ("prep", prep_unscaled),
            ("model", TabPFNClassifier())
        ]),
    }

    return models


# =========================================================
# Main workflow
# =========================================================
def main():
    # -----------------------------
    # 1) Load and inspect data
    # -----------------------------
    df = load_data(FILE, SHEET_NAME)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    print("Shape:", df.shape)
    print("Target distribution:\n", y.value_counts())

    # Save simple dataset summary
    summary_df = pd.DataFrame({
        "metric": ["n_rows", "n_columns", "n_sporadic_0", "n_familial_1"],
        "value": [df.shape[0], df.shape[1], int((y == 0).sum()), int((y == 1).sum())]
    })
    save_df(summary_df, "dataset_summary.csv")

    # -----------------------------
    # 2) Biological comparison:
    #    symptom differences by group
    # -----------------------------
    group_diff_df = get_group_difference_table(df, TARGET)
    print("\nTop symptom differences by class:")
    print(group_diff_df.head(10).to_string(index=False))
    save_df(group_diff_df, "group_symptom_differences.csv")

    # -----------------------------
    # 3) ANOVA feature ranking
    # -----------------------------
    anova_df = get_anova_table(X, y)
    print("\nANOVA feature ranking:")
    print(anova_df.head(10).to_string(index=False))
    save_df(anova_df, "anova_feature_ranking.csv")

    # -----------------------------
    # 4) Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 5) Model comparison
    # -----------------------------
    models = build_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "auc": "roc_auc"
    }

    rows = []
    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        )

        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1": scores["test_f1"].mean(),
            "cv_auc": scores["test_auc"].mean(),
        })

    results_df = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    print("\nCV Results:")
    print(results_df.to_string(index=False))
    save_df(results_df, "cv_results.csv")

    # -----------------------------
    # 6) Fit best model on train set
    #    and evaluate on test set
    # -----------------------------
    best_name = results_df.iloc[0]["model"]
    best_pipe = models[best_name]

    best_pipe.fit(X_train, y_train)

    pred = best_pipe.predict(X_test)
    proba = best_pipe.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, pred)
    test_f1 = f1_score(y_test, pred)
    test_auc = roc_auc_score(y_test, proba)

    print(f"\nBest model: {best_name}")
    print("TEST Accuracy:", test_accuracy)
    print("TEST F1:", test_f1)
    print("TEST AUC:", test_auc)
    print("\nClassification report:\n", classification_report(y_test, pred))

    test_results_df = pd.DataFrame({
        "best_model": [best_name],
        "test_accuracy": [test_accuracy],
        "test_f1": [test_f1],
        "test_auc": [test_auc]
    })
    save_df(test_results_df, "test_results.csv")

    # Save classification report as CSV-like table
    report_dict = classification_report(y_test, pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class_or_metric"})
    save_df(report_df, "classification_report.csv")

    # -----------------------------
    # 7) Permutation importance
    #    for biological interpretation
    # -----------------------------
    perm = permutation_importance(
        best_pipe,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc"
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    print("\nTop important features:")
    print(importance_df.head(10).to_string(index=False))
    save_df(importance_df, "permutation_importance.csv")

    # -----------------------------
    # 8) Optional: evaluate a simple
    #    model using top ANOVA features
    # -----------------------------
    top_k = 8
    top_features = anova_df.head(top_k)["feature"].tolist()

    X_top = X[top_features]
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
        X_top,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    top_feature_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    top_feature_model.fit(X_train_top, y_train_top)
    pred_top = top_feature_model.predict(X_test_top)
    proba_top = top_feature_model.predict_proba(X_test_top)[:, 1]

    top_feature_results = pd.DataFrame({
        "model": ["LogReg_top_ANOVA_features"],
        "num_features": [top_k],
        "features_used": [", ".join(top_features)],
        "test_accuracy": [accuracy_score(y_test_top, pred_top)],
        "test_f1": [f1_score(y_test_top, pred_top)],
        "test_auc": [roc_auc_score(y_test_top, proba_top)]
    })

    print("\nTop-feature Logistic Regression results:")
    print(top_feature_results.to_string(index=False))
    save_df(top_feature_results, "top_feature_model_results.csv")


if __name__ == "__main__":
    main()