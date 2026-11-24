import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
import trees


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    categorical: list


def tenure_group(months):
    for (low, high), label in zip(config.TENURE_BINS, config.TENURE_LABELS):
        if low <= months <= high:
            return label
    return config.TENURE_LABELS[-1]


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = total.isna()
    # Baris kosong hanya milik pelanggan baru (tenure 0), jadi total tagihannya memang 0
    assert (df.loc[blank, "tenure"] == 0).all()
    df["TotalCharges"] = total.fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns=[config.ID_COLUMN]), int(blank.sum())


def build_pipeline(categorical, n_estimators, max_depth):
    encoder = ColumnTransformer(
        [("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", encoder), ("forest", forest)])


def fit_and_score(split, n_estimators, max_depth):
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    model = build_pipeline(split.categorical, n_estimators, max_depth)
    cv_scores = cross_validate(model, split.X_train, split.y_train, cv=cv, scoring={"accuracy": "accuracy", "f1": "f1"})
    model.fit(split.X_train, split.y_train)
    pred = model.predict(split.X_test)
    proba = model.predict_proba(split.X_test)[:, 1]
    row = {
        "train_acc": accuracy_score(split.y_train, model.predict(split.X_train)),
        "test_acc": accuracy_score(split.y_test, pred),
        "cv_acc": cv_scores["test_accuracy"].mean(),
        "cv_std": cv_scores["test_accuracy"].std(),
        "cv_f1": cv_scores["test_f1"].mean(),
        "test_recall": recall_score(split.y_test, pred),
        "test_f1": f1_score(split.y_test, pred),
        "test_auc": roc_auc_score(split.y_test, proba),
    }
    return model, {key: round(float(val), 4) for key, val in row.items()}


def choose_simplest(rows):
    # F1 kelas churn dipakai karena akurasi menyesatkan pada data yang tidak seimbang
    best = max(row["cv_f1"] for row in rows)
    return next(row for row in rows if row["cv_f1"] >= best - config.SELECTION_TOLERANCE)


def run_scenario_a(split):
    rows = []
    for n in config.N_ESTIMATORS_GRID:
        print(f"Skenario A: n_estimators={n}")
        _, row = fit_and_score(split, n, None)
        row["value"] = n
        row["label"] = str(n)
        rows.append(row)
    return {"rows": rows, "chosen": choose_simplest(rows)["value"]}


def run_scenario_b(split, n_estimators):
    rows = []
    for depth in config.MAX_DEPTH_GRID:
        print(f"Skenario B: max_depth={depth}")
        _, row = fit_and_score(split, n_estimators, depth)
        row["value"] = depth
        row["label"] = "Tak terbatas" if depth is None else str(depth)
        rows.append(row)
    return {"rows": rows, "chosen": choose_simplest(rows)["value"], "n_estimators": n_estimators}


def split_feature_name(name, categorical):
    for column in categorical:
        if name.startswith(column + "_"):
            return column, name[len(column) + 1:]
    return name, None


def feature_label(column, value):
    label = config.FEATURE_LABELS.get(column, column)
    if value is None:
        return label
    return f"{label}: {config.VALUE_LABELS.get(value, value)}"


def importance_table(model, categorical):
    names = model.named_steps["prep"].get_feature_names_out()
    scores = model.named_steps["forest"].feature_importances_
    table = []
    for name, score in zip(names, scores):
        column, value = split_feature_name(name, categorical)
        table.append(
            {
                "name": name,
                "column": column,
                "value": value,
                "label": feature_label(column, value),
                "importance": round(float(score), 4),
            }
        )
    return sorted(table, key=lambda item: item["importance"], reverse=True)


def churn_rates(df):
    y = (df[config.TARGET] == "Yes").astype(int)
    grouped = df.assign(TenureGroup=df["tenure"].map(tenure_group), _y=y)
    rates = {}
    for column in [c for c in df.columns if c not in config.NUMERIC_FEATURES + [config.TARGET]] + ["TenureGroup"]:
        stats = grouped.groupby(column)["_y"].agg(["mean", "size"])
        rates[column] = {
            str(value): {"rate": round(float(row["mean"]), 4), "n": int(row["size"])}
            for value, row in stats.iterrows()
        }
    return rates


def final_report(model, split):
    pred = model.predict(split.X_test)
    proba = model.predict_proba(split.X_test)[:, 1]
    report = classification_report(split.y_test, pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(split.y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(split.y_test, proba)

    def clean(entry):
        return {key: round(float(val), 4) if key != "support" else int(val) for key, val in entry.items()}

    return {
        "report": {
            "stay": clean(report["0"]),
            "churn": clean(report["1"]),
            "macro": clean(report["macro avg"]),
            "weighted": clean(report["weighted avg"]),
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "auc": round(float(roc_auc_score(split.y_test, proba)), 4),
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
    }


def run():
    df, n_blank = load_data()
    X, y = df.drop(columns=[config.TARGET]), (df[config.TARGET] == "Yes").astype(int)
    categorical = [c for c in X.columns if c not in config.NUMERIC_FEATURES]
    split = Split(*train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE), categorical)

    scenario_a = run_scenario_a(split)
    scenario_b = run_scenario_b(split, scenario_a["chosen"])
    depth = scenario_b["chosen"]

    print(f"Model akhir: n_estimators={scenario_a['chosen']}, max_depth={depth}")
    final_row = next(r for r in scenario_b["rows"] if r["value"] == depth)
    model = build_pipeline(categorical, scenario_a["chosen"], depth).fit(split.X_train, split.y_train)

    metrics = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "n_rows": len(df),
            "n_features": X.shape[1],
            "n_train": len(split.X_train),
            "n_test": len(split.X_test),
            "n_blank_total": n_blank,
            "n_churn": int(y.sum()),
            "n_stay": int((1 - y).sum()),
            "churn_rate": round(float(y.mean()), 4),
        },
        "feature_order": list(X.columns),
        "categorical": categorical,
        "choices": {c: sorted(X[c].unique().tolist()) for c in categorical},
        "defaults": {
            **{c: X[c].mode()[0] for c in categorical},
            **{c: round(float(X[c].median()), 2) for c in config.NUMERIC_FEATURES},
        },
        "category_churn": churn_rates(df),
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "final": {
            "n_estimators": scenario_a["chosen"],
            "max_depth": depth,
            "train_acc": final_row["train_acc"],
            "test_acc": final_row["test_acc"],
            "cv_acc": final_row["cv_acc"],
            "cv_std": final_row["cv_std"],
            "cv_f1": final_row["cv_f1"],
            "importances": importance_table(model, categorical),
            **final_report(model, split),
        },
    }

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH, compress=3)
    config.METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import figures  # matplotlib hanya dibutuhkan saat melatih ulang
    except ImportError:
        print("matplotlib tidak terpasang, gambar tidak dibuat ulang.")
    else:
        figures.make_all(metrics)
        figures.tree_example(trees.example_tree(model, metrics["choices"]))
    return model, metrics


if __name__ == "__main__":
    run()
