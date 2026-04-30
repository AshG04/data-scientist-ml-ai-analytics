import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_FILE = "synthetic_customer_case_data.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["case_month"] = pd.to_datetime(df["case_month"], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_fields = [
        "case_id", "case_month", "customer_segment", "channel",
        "case_type", "priority", "case_age_days", "touchpoints",
        "sla_breach_flag", "resolution_time_hours", "escalated_flag"
    ]

    clean = df.dropna(subset=required_fields).copy()
    clean = clean[clean["case_age_days"] >= 0]
    clean = clean[clean["touchpoints"] >= 0]
    clean = clean[clean["resolution_time_hours"] >= 0]

    return clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()

    model_df["high_touch_flag"] = (model_df["touchpoints"] >= 5).astype(int)
    model_df["aged_case_flag"] = (model_df["case_age_days"] >= 10).astype(int)
    model_df["long_resolution_flag"] = (model_df["resolution_time_hours"] >= 72).astype(int)

    categorical_cols = ["customer_segment", "channel", "case_type", "priority"]
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

    return model_df


def train_model(model_df: pd.DataFrame):
    target = "escalated_flag"
    excluded = ["case_id", "case_month", target, "experiment_group"]
    feature_cols = [col for col in model_df.columns if col not in excluded]

    X = model_df[feature_cols]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame([
        {
            "model_name": "RandomForestClassifier",
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3),
            "f1_score": round(f1_score(y_test, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
            "test_records": len(y_test)
        }
    ])

    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return model, feature_cols, metrics, feature_importance


def create_model_monitoring(df: pd.DataFrame) -> pd.DataFrame:
    monitoring = (
        df.groupby("case_month", as_index=False)
        .agg(
            total_cases=("case_id", "nunique"),
            escalation_rate=("escalated_flag", "mean"),
            sla_breach_rate=("sla_breach_flag", "mean"),
            avg_resolution_time_hours=("resolution_time_hours", "mean"),
            avg_touchpoints=("touchpoints", "mean"),
            avg_case_age_days=("case_age_days", "mean")
        )
    )

    monitoring["escalation_rate"] = monitoring["escalation_rate"].round(3)
    monitoring["sla_breach_rate"] = monitoring["sla_breach_rate"].round(3)
    monitoring["avg_resolution_time_hours"] = monitoring["avg_resolution_time_hours"].round(1)
    monitoring["avg_touchpoints"] = monitoring["avg_touchpoints"].round(1)
    monitoring["avg_case_age_days"] = monitoring["avg_case_age_days"].round(1)

    return monitoring


def analyze_ab_test(df: pd.DataFrame) -> pd.DataFrame:
    ab = (
        df.groupby("experiment_group", as_index=False)
        .agg(
            total_cases=("case_id", "nunique"),
            escalation_rate=("escalated_flag", "mean"),
            sla_breach_rate=("sla_breach_flag", "mean"),
            avg_resolution_time_hours=("resolution_time_hours", "mean")
        )
    )

    ab["escalation_rate"] = ab["escalation_rate"].round(3)
    ab["sla_breach_rate"] = ab["sla_breach_rate"].round(3)
    ab["avg_resolution_time_hours"] = ab["avg_resolution_time_hours"].round(1)

    baseline_rate = ab.loc[ab["experiment_group"] == "Control", "escalation_rate"].iloc[0]
    ab["lift_vs_control"] = ((baseline_rate - ab["escalation_rate"]) / baseline_rate).round(3)

    return ab


def main():
    df = load_data(DATA_FILE)
    df = clean_data(df)
    model_df = engineer_features(df)

    model, feature_cols, metrics, feature_importance = train_model(model_df)
    monitoring = create_model_monitoring(df)
    ab_summary = analyze_ab_test(df)

    metrics.to_csv(OUTPUT_DIR / "model_evaluation_output.csv", index=False)
    feature_importance.head(20).to_csv(OUTPUT_DIR / "feature_importance_output.csv", index=False)
    monitoring.to_csv(OUTPUT_DIR / "model_monitoring_output.csv", index=False)
    ab_summary.to_csv(OUTPUT_DIR / "ab_test_summary_output.csv", index=False)

    print("ML risk prediction pipeline completed successfully.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
