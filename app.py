import json

import joblib
import pandas as pd
import sklearn
from flask import Flask, render_template, request

import config
import insights
import train_model
import trees
import figures

app = Flask(__name__)

NAV = [
    ("index", "Beranda", "home"),
    ("predict", "Prediksi", "predict"),
    ("dataset", "Dataset", "data"),
    ("experiments", "Eksperimen", "lab"),
    ("model_page", "Model", "model"),
]


def load_artifacts():
    if config.MODEL_PATH.exists() and config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        # Model hasil joblib hanya aman dimuat pada versi scikit-learn yang sama
        if metrics.get("sklearn_version") == sklearn.__version__:
            return joblib.load(config.MODEL_PATH), metrics
    return train_model.run()


MODEL, METRICS = load_artifacts()
TEXTS = insights.build_all(METRICS)
TREE_EXAMPLE = trees.example_tree(MODEL, METRICS["choices"])
TEXTS["model"]["tree"] = insights.tree_example_text(TREE_EXAMPLE)
if not (config.IMG_DIR / "tree_example.svg").exists():
    figures.tree_example(TREE_EXAMPLE)

app.jinja_env.filters.update(
    pct=insights.pct,
    pts=insights.pts,
    dec=insights.dec,
    num=insights.num,
    depth_text=insights.depth_text,
    value_text=insights.value_text,
)


@app.context_processor
def shared_context():
    return {
        "nav": NAV,
        "m": METRICS,
        "version": METRICS["trained_at"],
        "labels": config.FEATURE_LABELS,
        "threshold": config.THRESHOLD,
    }


def parse_number(raw, label, cast):
    low, high = config.INPUT_LIMITS[label]
    try:
        number = cast(raw.replace(",", "."))
    except ValueError:
        return None, f"{config.FEATURE_LABELS[label]} harus berupa angka yang valid."
    if not low <= number <= high:
        return None, f"{config.FEATURE_LABELS[label]} harus antara {low} dan {high}."
    return number, None


def normalize_services(values):
    no_internet = values["InternetService"] == "No"
    for column in config.INTERNET_ADDONS:
        if no_internet:
            values[column] = "No internet service"
        elif values[column] == "No internet service":
            values[column] = "No"
    if values["PhoneService"] == "No":
        values["MultipleLines"] = "No phone service"
    elif values["MultipleLines"] == "No phone service":
        values["MultipleLines"] = "No"


def read_customer(form):
    values, errors = {}, []
    for column in METRICS["categorical"]:
        value = form.get(column, "")
        if value not in METRICS["choices"][column]:
            errors.append(f"Pilihan {config.FEATURE_LABELS[column].lower()} tidak valid.")
        values[column] = value
    for column, cast in (("tenure", int), ("MonthlyCharges", float)):
        values[column], error = parse_number(form.get(column, "").strip(), column, cast)
        if error:
            errors.append(error)
    raw_total = form.get("TotalCharges", "").strip()
    if raw_total:
        values["TotalCharges"], error = parse_number(raw_total, "TotalCharges", float)
        if error:
            errors.append(error)
    elif not errors:
        # Total tagihan dikosongkan: dihitung kasar dari lama berlangganan dikali tagihan bulanan
        values["TotalCharges"] = round(values["tenure"] * values["MonthlyCharges"], 2)
    if errors:
        return None, errors
    normalize_services(values)
    return values, []


def profile_factors(values):
    row = {**values, "TenureGroup": train_model.tenure_group(values["tenure"])}
    factors = []
    for column in config.KEY_FACTORS:
        raw = row[column]
        stats = METRICS["category_churn"][column][raw]
        factors.append(
            {
                "column": column,
                "raw": raw,
                "label": config.FEATURE_LABELS[column],
                "value": config.VALUE_LABELS.get(raw, raw),
                "rate": stats["rate"],
                "n": stats["n"],
            }
        )
    return factors


def predict_customer(values):
    frame = pd.DataFrame([values])[METRICS["feature_order"]]
    score = float(MODEL.predict_proba(frame)[0, 1])
    factors = profile_factors(values)
    walk = trees.forest_walk(MODEL, frame, METRICS["choices"])
    return {
        "score": score,
        "walk": walk,
        **insights.explain_walk(walk, score),
        "flagged": score >= config.THRESHOLD,
        "factors": factors,
        **insights.explain_prediction(score, factors, METRICS),
    }


def default_form():
    defaults = METRICS["defaults"]
    form = {column: defaults[column] for column in METRICS["categorical"]}
    form.update(tenure=str(int(defaults["tenure"])), MonthlyCharges=str(defaults["MonthlyCharges"]), TotalCharges="")
    return form


@app.route("/")
def index():
    return render_template("index.html", summary=TEXTS["home"])


@app.route("/dataset")
def dataset():
    return render_template("dataset.html", text=TEXTS["dataset"])


@app.route("/eksperimen")
def experiments():
    return render_template("experiments.html", text=TEXTS["experiments"])


@app.route("/model")
def model_page():
    return render_template("model.html", text=TEXTS["model"])


@app.route("/prediksi", methods=["GET", "POST"])
def predict():
    form, result, errors = default_form(), None, []
    if request.method == "POST":
        form = {key: request.form.get(key, "") for key in METRICS["feature_order"]}
        values, errors = read_customer(request.form)
        if values:
            result = predict_customer(values)
    return render_template(
        "predict.html",
        form=form,
        result=result,
        errors=errors,
        groups=config.FORM_GROUPS,
        choices=METRICS["choices"],
        presets=config.PRESETS,
    )


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run()
