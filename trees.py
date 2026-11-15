import numpy as np

import config


def fmt(value):
    text = f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


class Splits:
    """Mengubah split pohon (indeks fitur dan ambang) menjadi pertanyaan ya atau tidak."""

    def __init__(self, model, choices):
        prep = model.named_steps["prep"]
        self.model = model
        self.prep = prep
        self.forest = model.named_steps["forest"]
        self.churn_index = list(self.forest.classes_).index(1)
        self.choices = choices
        self.columns = []
        categorical = list(choices)
        for name in prep.get_feature_names_out():
            column = next((c for c in categorical if name.startswith(f"{c}_")), None)
            self.columns.append((column, name[len(column) + 1 :]) if column else (name, None))

    def question(self, feature, threshold):
        column, value = self.columns[feature]
        label = config.FEATURE_LABELS[column]
        if value is None:
            unit = " bulan" if column == "tenure" else ""
            return f"{label.split(' (')[0]} > {fmt(threshold)}{unit}"
        return f"{label}: {config.VALUE_LABELS.get(value, value)}"

    def leaf_score(self, tree, node):
        counts = tree.value[node, 0]
        return float(counts[self.churn_index] / counts.sum())


def forest_walk(model, frame, choices):
    """Skor tiap pohon untuk satu pelanggan, dan jalur pohon yang paling dekat dengan skor hutan."""
    splits = Splits(model, choices)
    X = splits.prep.transform(frame).astype(np.float32)
    scores = [float(tree.predict_proba(X)[0, splits.churn_index]) for tree in splits.forest.estimators_]
    mean = float(np.mean(scores))
    picked = int(np.argmin([abs(score - mean) for score in scores]))
    tree = splits.forest.estimators_[picked]
    nodes = tree.decision_path(X).indices
    steps = []
    for node in nodes[:-1]:
        feature, threshold = int(tree.tree_.feature[node]), float(tree.tree_.threshold[node])
        value = float(X[0, feature])
        column, category = splits.columns[feature]
        steps.append(
            {
                "question": splits.question(feature, threshold),
                "yes": value > threshold,
                "shown": fmt(value) if category is None else "",
            }
        )
    leaf = int(nodes[-1])
    return {
        "scores": scores,
        "n_trees": len(scores),
        "n_churn": sum(score >= config.THRESHOLD for score in scores),
        "mean": mean,
        "picked": picked,
        "tree_score": splits.leaf_score(tree.tree_, leaf),
        "leaf_n": int(tree.tree_.n_node_samples[leaf]),
        "steps": steps,
    }


def example_tree(model, choices, levels=2):
    """Pohon pertama dari hutan, dipotong pada beberapa tingkat teratas untuk digambar."""
    splits = Splits(model, choices)
    estimator = splits.forest.estimators_[0]
    tree = estimator.tree_

    def build(node, depth, trail):
        data = {
            "score": splits.leaf_score(tree, node),
            "n": int(tree.n_node_samples[node]),
            "trail": trail,
            "leaf": tree.children_left[node] == -1,
        }
        if data["leaf"] or depth == levels:
            data["leaf"] = True
            return data
        question = splits.question(int(tree.feature[node]), float(tree.threshold[node]))
        data["question"] = question
        data["no"] = build(tree.children_left[node], depth + 1, trail + [(question, False)])
        data["yes"] = build(tree.children_right[node], depth + 1, trail + [(question, True)])
        return data

    return {"root": build(0, 0, []), "depth": int(estimator.get_depth()), "levels": levels, "n_trees": len(splits.forest.estimators_)}
