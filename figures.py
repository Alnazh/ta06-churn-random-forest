import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import config

P = config.PALETTE
SIZE = (6, 4)

plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "svg.hashsalt": "churnforest",
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 12,
        "text.color": P["ink"],
        "axes.labelcolor": P["ink"],
        "xtick.color": P["ink"],
        "ytick.color": P["ink"],
        "axes.edgecolor": P["ink"],
        "axes.linewidth": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#d9d2bd",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.transparent": True,
    }
)


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def num(value):
    return f"{value:,}".replace(",", ".")


def save(fig, name):
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.IMG_DIR / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def class_distribution(m):
    data = m["dataset"]
    counts = [data["n_stay"], data["n_churn"]]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(["Bertahan", "Churn"], counts, color=[P["blue"], P["red"]], width=0.55, edgecolor=P["ink"], linewidth=2)
    for x, count in enumerate(counts):
        ax.text(x, count, f"{num(count)}\n{pct(count / data['n_rows'])}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_ylabel("Jumlah pelanggan")
    save(fig, "class_distribution")


def churn_by_group(m, column, order, labels, name):
    stats = m["category_churn"][column]
    base = m["dataset"]["churn_rate"]
    rates = [stats[key]["rate"] for key in order]
    fig, ax = plt.subplots(figsize=SIZE)
    colors = [P["red"] if rate > base else P["blue"] for rate in rates]
    ax.bar(labels, rates, color=colors, width=0.55, edgecolor=P["ink"], linewidth=2)
    ax.axhline(base, color=P["ink"], linestyle="--", linewidth=1.2)
    ax.text(len(order) - 0.5, base, f" rata-rata {pct(base)}", va="bottom", ha="right", fontsize=11)
    for x, rate in enumerate(rates):
        ax.text(x, rate, pct(rate), ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.2)
    ax.set_ylabel("Tingkat churn")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    save(fig, name)


def churn_by_contract(m):
    order = ["Month-to-month", "One year", "Two year"]
    churn_by_group(m, "Contract", order, [config.VALUE_LABELS[k] for k in order], "churn_by_contract")


def churn_by_tenure(m):
    churn_by_group(m, "TenureGroup", config.TENURE_LABELS, config.TENURE_LABELS, "churn_by_tenure")


def highlight_chosen(ax, labels, chosen_label):
    index = labels.index(chosen_label)
    ax.axvspan(index - 0.35, index + 0.35, color=P["yellow"], alpha=0.3, linewidth=0)


def scenario_a(m):
    rows = m["scenario_a"]["rows"]
    labels = [r["label"] for r in rows]
    test = [r["test_acc"] for r in rows]
    cv = [r["cv_acc"] for r in rows]
    fig, ax = plt.subplots(figsize=SIZE)
    highlight_chosen(ax, labels, str(m["scenario_a"]["chosen"]))
    ax.plot(labels, test, color=P["blue"], marker="o", linewidth=2.4, markersize=8, label="Akurasi uji")
    ax.plot(labels, cv, color=P["ink"], marker="s", linestyle="--", linewidth=1.6, markersize=6, label="Validasi silang")
    for x, value in enumerate(test):
        ax.annotate(pct(value), (x, value), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11, color=P["blue"], fontweight="bold")
    low, high = min(test + cv), max(test + cv)
    pad = max((high - low) * 0.6, 0.01)
    ax.set_ylim(low - pad, high + pad)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.1f}%")
    ax.set_xlabel("Jumlah pohon (n_estimators)")
    ax.set_ylabel("Akurasi")
    ax.margins(x=0.1)
    ax.legend(loc="lower right")
    save(fig, "scenario_a")


def scenario_b(m):
    rows = m["scenario_b"]["rows"]
    labels = [r["label"] for r in rows]
    train = [r["train_acc"] for r in rows]
    test = [r["test_acc"] for r in rows]
    chosen = m["scenario_b"]["chosen"]
    fig, ax = plt.subplots(figsize=SIZE)
    highlight_chosen(ax, labels, "Tak terbatas" if chosen is None else str(chosen))
    ax.fill_between(labels, test, train, color=P["red"], alpha=0.12, linewidth=0)
    ax.plot(labels, train, color=P["red"], marker="o", linewidth=2.4, markersize=8, label="Akurasi latih")
    ax.plot(labels, test, color=P["blue"], marker="o", linewidth=2.4, markersize=8, label="Akurasi uji")
    for x, (tr, te) in enumerate(zip(train, test)):
        ax.annotate(pct(tr), (x, tr), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11, color=P["red"], fontweight="bold")
        ax.annotate(pct(te), (x, te), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=11, color=P["blue"], fontweight="bold")
    low, high = min(train + test), max(train + test)
    pad = (high - low) * 0.18
    ax.set_ylim(low - pad, high + pad)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_xlabel("Kedalaman pohon (max_depth)")
    ax.set_ylabel("Akurasi")
    ax.margins(x=0.12)
    ax.legend(loc="upper left")
    save(fig, "scenario_b")


def feature_importance(m):
    top = m["final"]["importances"][:10][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    colors = [P["red"] if i >= len(top) - 3 else P["blue"] for i in range(len(top))]
    ax.barh([textwrap.fill(t["label"], 22) for t in top], [t["importance"] for t in top], color=colors, height=0.62, edgecolor=P["ink"], linewidth=2)
    for y, item in enumerate(top):
        ax.text(item["importance"], y, f" {pct(item['importance'])}", va="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, top[-1]["importance"] * 1.18)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_xlabel("Tingkat kepentingan")
    save(fig, "feature_importance")


def confusion_matrix_chart(m):
    c = m["final"]["confusion"]
    cells = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for row in range(2):
        total = sum(cells[row])
        for col in range(2):
            share = cells[row][col] / total
            correct = row == col
            alpha = 0.2 + 0.8 * share if correct else 0.15 + 0.85 * share
            ax.add_patch(Rectangle((col, row), 1, 1, color=P["blue"] if correct else P["red"], alpha=alpha, linewidth=0))
            ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor=P["ink"], linewidth=2))
            light = alpha > 0.55
            ax.text(col + 0.5, row + 0.45, num(cells[row][col]), ha="center", va="center", fontsize=22, fontweight="bold", color="white" if light else P["ink"])
            ax.text(col + 0.5, row + 0.72, f"{pct(share)} dari baris", ha="center", va="center", fontsize=11, color="white" if light else P["ink"])
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5], ["Bertahan", "Churn"])
    ax.set_yticks([0.5, 1.5], ["Bertahan", "Churn"])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Kenyataan")
    save(fig, "confusion_matrix")


def roc_curve_chart(m):
    roc = m["final"]["roc"]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot([0, 1], [0, 1], color=P["grey"], linestyle="--", linewidth=1.4, label="Tebakan acak")
    ax.fill_between(roc["fpr"], roc["tpr"], color=P["blue"], alpha=0.12, linewidth=0)
    ax.plot(roc["fpr"], roc["tpr"], color=P["blue"], linewidth=2.6, label=f"Random Forest (AUC {m['final']['auc']:.2f})".replace(".", ","))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Salah tandai (False Positive Rate)")
    ax.set_ylabel("Churn tertangkap (True Positive Rate)")
    ax.legend(loc="lower right")
    save(fig, "roc_curve")


def tree_example(t):
    levels = t["levels"]
    ys = {d: 0.88 - d * 0.74 / levels for d in range(levels + 1)}
    box_w, box_h = 0.24, 0.2

    def place(node, depth, x, span):
        node["x"], node["y"] = x, ys[depth]
        if not node["leaf"]:
            place(node["no"], depth + 1, x - span / 4, span / 2)
            place(node["yes"], depth + 1, x + span / 4, span / 2)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    place(t["root"], 0, 0.5, 1.0)

    def draw_edges(node):
        if node["leaf"]:
            return
        for key, label in (("no", "Tidak"), ("yes", "Ya")):
            child = node[key]
            ax.plot([node["x"], child["x"]], [node["y"] - box_h / 2, child["y"] + box_h / 2], color=P["ink"], linewidth=2, zorder=1)
            ax.text(
                (node["x"] + child["x"]) / 2, (node["y"] + child["y"]) / 2, label, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=3,
                bbox={"boxstyle": "round,pad=0.25", "fc": P["yellow"], "ec": P["ink"], "lw": 1.6},
            )
            draw_edges(child)

    def draw_boxes(node):
        if node["leaf"]:
            color = P["red"] if node["score"] >= config.THRESHOLD else P["blue"]
            text = f"Churn {pct(node['score'], 0)}\nn = {num(node['n'])}"
            weight = "bold"
        else:
            color = "white"
            text = textwrap.fill(node["question"], 15)
            weight = "normal"
        ax.add_patch(FancyBboxPatch((node["x"] - box_w / 2, node["y"] - box_h / 2), box_w, box_h, boxstyle="round,pad=0,rounding_size=0.015", fc=color, ec=P["ink"], lw=2, zorder=2))
        ax.text(node["x"], node["y"], text, ha="center", va="center", fontsize=9.6, fontweight=weight, zorder=4, color="white" if color == P["blue"] else P["ink"])
        if not node["leaf"]:
            draw_boxes(node["no"])
            draw_boxes(node["yes"])

    draw_edges(t["root"])
    draw_boxes(t["root"])
    save(fig, "tree_example")


def make_all(m):
    for draw in (
        class_distribution,
        churn_by_contract,
        churn_by_tenure,
        scenario_a,
        scenario_b,
        feature_importance,
        confusion_matrix_chart,
        roc_curve_chart,
    ):
        draw(m)
