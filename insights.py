from collections import defaultdict

import config


def dec(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1):
    return dec(value * 100, digits) + "%"


def pts(value, digits=1):
    return dec(abs(value) * 100, digits)


def num(value):
    return f"{value:,}".replace(",", ".")


def depth_text(depth):
    return "tak terbatas" if depth is None else str(depth)


def short(label):
    return label.split(" (")[0]


def value_text(value):
    return config.VALUE_LABELS.get(value, value)


def join_list(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " dan " + items[-1]


def rate_of(m, column, value):
    return m["category_churn"].get(column, {}).get(value)


def dataset_class(m):
    d = m["dataset"]
    stay_rate = 1 - d["churn_rate"]
    ratio = d["n_stay"] / d["n_churn"]
    lines = [
        f"Dari {num(d['n_rows'])} pelanggan, {num(d['n_churn'])} ({pct(d['churn_rate'])}) berhenti berlangganan "
        f"dan {num(d['n_stay'])} ({pct(stay_rate)}) bertahan. Perbandingannya sekitar 1 banding {dec(ratio, 1)}."
    ]
    if ratio >= 1.5:
        lines.append(
            f"Kelasnya tidak seimbang. Model yang menebak semua pelanggan bertahan akan benar {pct(stay_rate)} "
            "tanpa menangkap satu pun pelanggan churn, jadi akurasi saja tidak cukup untuk menilai model. "
            "Karena itu model dilatih dengan class_weight balanced dan dinilai juga dengan recall serta F1-score kelas churn."
        )
    else:
        lines.append("Kelasnya relatif seimbang, jadi akurasi sudah cukup mewakili kinerja model.")
    return lines


def dataset_contract(m):
    stats = m["category_churn"]["Contract"]
    ranked = sorted(stats.items(), key=lambda item: item[1]["rate"], reverse=True)
    (top_key, top), (low_key, low) = ranked[0], ranked[-1]
    lines = [
        f"Pelanggan dengan kontrak {value_text(top_key).lower()} churn {pct(top['rate'])} ({num(top['n'])} pelanggan), "
        f"sedangkan kontrak {value_text(low_key).lower()} hanya {pct(low['rate'])}. "
        f"Selisihnya sekitar {dec(top['rate'] / low['rate'], 1)} kali lipat."
    ]
    order = ["Month-to-month", "One year", "Two year"]
    rates = [stats[key]["rate"] for key in order]
    if rates == sorted(rates, reverse=True):
        lines.append("Semakin panjang masa kontrak, semakin rendah churn. Ini hubungan yang terlihat di data dan belum tentu sebab akibat.")
    else:
        lines.append("Pola churn tidak turun rapi mengikuti panjang kontrak, jadi hubungan ini perlu dicek lebih lanjut.")
    return lines


def dataset_tenure(m):
    stats = m["category_churn"]["TenureGroup"]
    labels = config.TENURE_LABELS
    rates = [stats[label]["rate"] for label in labels]
    lines = [
        f"Kelompok {labels[0]} churn {pct(rates[0])}, sedangkan kelompok {labels[-1]} hanya {pct(rates[-1])}."
    ]
    if rates == sorted(rates, reverse=True):
        lines.append("Churn turun di setiap kelompok yang lebih lama, jadi masa paling rawan ada di bulan-bulan awal berlangganan.")
    else:
        lines.append("Churn tidak turun konsisten mengikuti lama berlangganan.")
    return lines


def dataset_missing(m):
    n = m["dataset"]["n_blank_total"]
    return [
        f"Kolom TotalCharges punya {n} baris kosong. Semuanya milik pelanggan dengan tenure 0 yang belum pernah ditagih, "
        "jadi nilainya diisi 0 dan barisnya dipertahankan. Kolom customerID dibuang karena hanya pengenal."
    ]


def scenario_a(m):
    rows = m["scenario_a"]["rows"]
    chosen = next(r for r in rows if r["value"] == m["scenario_a"]["chosen"])
    test = [r["test_acc"] for r in rows]
    spread = max(test) - min(test)
    best_test = max(rows, key=lambda r: r["test_acc"])
    plateau = next(r for r in rows if r["test_acc"] >= max(test) - 0.005)
    late = rows[-1]["test_acc"] - rows[2]["test_acc"]
    n_test = m["dataset"]["n_test"]

    lines = [
        f"Akurasi uji berada di rentang {pct(min(test))} sampai {pct(max(test))} untuk semua jumlah pohon, "
        f"selisih terbesarnya {pts(spread)} poin persentase."
    ]
    if abs(late) < 0.005:
        lines.append(f"Dari {rows[2]['label']} ke {rows[-1]['label']} pohon akurasi hampir tidak berubah ({pts(late)} poin).")
    else:
        lines.append(f"Dari {rows[2]['label']} ke {rows[-1]['label']} pohon akurasi {'naik' if late > 0 else 'turun'} {pts(late)} poin.")
    lines.append(
        f"Data uji berisi {num(n_test)} pelanggan, jadi satu pelanggan saja mengubah akurasi {dec(100 / n_test, 2)} poin. "
        "Selisih yang setara beberapa pelanggan sebaiknya dianggap kebetulan."
    )
    if spread < 0.01:
        lines.append(f"Menambah pohon terus-menerus tidak meningkatkan akurasi secara signifikan. Akurasi sudah mendatar sejak sekitar {plateau['label']} pohon.")
    else:
        lines.append(f"Penambahan pohon memberi perbaikan nyata, tetapi akurasi mulai mendatar sekitar {plateau['label']} pohon.")
    lines.append(
        f"Pilihan akhir dibuat dengan F1-score churn pada validasi silang {config.CV_FOLDS} lipatan, bukan dengan akurasi uji, "
        f"agar data uji tidak ikut menentukan parameter. Jumlah pohon paling sedikit yang F1-nya masih dalam {pts(config.SELECTION_TOLERANCE)} poin "
        f"dari yang terbaik adalah {chosen['label']} pohon (F1 validasi silang {dec(chosen['cv_f1'])})."
    )
    if best_test["value"] != chosen["value"]:
        lines.append(
            f"Akurasi uji tertinggi memang ada di {best_test['label']} pohon ({pct(best_test['test_acc'])}), "
            f"tetapi F1 churn validasi silangnya {dec(best_test['cv_f1'])} dibanding {dec(chosen['cv_f1'])} pada pilihan akhir."
        )
    return lines


def scenario_b(m):
    rows = m["scenario_b"]["rows"]
    chosen = next(r for r in rows if r["value"] == m["scenario_b"]["chosen"])
    baseline = 1 - m["dataset"]["churn_rate"]

    def gap(row):
        return row["train_acc"] - row["test_acc"]

    onset = next((r for r in rows if gap(r) > config.OVERFIT_GAP), None)
    shallow_low = next((r for r in rows if r["test_acc"] < baseline), None)
    best_recall = max(rows, key=lambda r: r["test_recall"])
    deepest = rows[-1]

    lines = []
    if onset:
        lines.append(
            f"Selisih akurasi latih dan uji melebar menjadi {pts(gap(onset))} poin pada kedalaman {onset['label']}, "
            f"jadi di titik inilah model mulai overfitting (batas yang dipakai {pts(config.OVERFIT_GAP, 0)} poin)."
        )
    else:
        lines.append(f"Tidak ada kedalaman yang selisih latih-ujinya melewati {pts(config.OVERFIT_GAP, 0)} poin, jadi overfitting tidak terlihat pada akurasi.")
    if shallow_low:
        lines.append(
            f"Akurasi uji pada kedalaman {shallow_low['label']} ({pct(shallow_low['test_acc'])}) bahkan di bawah tebakan semua bertahan ({pct(baseline)}). "
            f"Itu terlihat buruk, tetapi model tersebut menangkap {pct(shallow_low['test_recall'])} pelanggan churn, sedangkan tebakan semua bertahan menangkap 0%."
        )
    drop = best_recall["test_recall"] - deepest["test_recall"]
    if drop > 0.05:
        lines.append(
            f"Akurasi uji tidak turun saat pohon makin dalam, tetapi recall churn turun dari {pct(best_recall['test_recall'])} "
            f"pada kedalaman {best_recall['label']} menjadi {pct(deepest['test_recall'])} tanpa batas kedalaman. "
            "Pohon yang terlalu dalam menghafal pola kelas mayoritas, jadi overfitting di sini paling terasa pada kelas churn."
        )
    else:
        lines.append(f"Recall churn relatif stabil di semua kedalaman (tertinggi {pct(best_recall['test_recall'])}).")
    lines.append(
        f"Kedalaman {chosen['label']} dipilih karena F1 churn validasi silangnya tertinggi atau setara dengan yang tertinggi ({dec(chosen['cv_f1'])}). "
        f"Selisih akurasi latih dan ujinya {pts(gap(chosen))} poin, recall churn {pct(chosen['test_recall'])}."
    )
    if gap(chosen) > config.OVERFIT_GAP:
        lines.append("Selisih ini masih di atas batas, jadi model terpilih perlu dibaca dengan hati-hati.")
    return lines


def grouped_importance(m):
    totals = defaultdict(float)
    for item in m["final"]["importances"]:
        totals[item["column"]] += item["importance"]
    return sorted(totals.items(), key=lambda pair: pair[1], reverse=True)


def explain_feature(item, m):
    column, value, label = item["column"], item["value"], short(item["label"])
    base = m["dataset"]["churn_rate"]
    if column == "tenure":
        groups = m["category_churn"]["TenureGroup"]
        first, last = groups[config.TENURE_LABELS[0]]["rate"], groups[config.TENURE_LABELS[-1]]["rate"]
        if first > last:
            return f"{label}. Pelanggan baru lebih mudah pergi. Di data, kelompok {config.TENURE_LABELS[0]} churn {pct(first)} dan kelompok {config.TENURE_LABELS[-1]} hanya {pct(last)}."
        return f"{label}. Di data, churn tidak lebih tinggi pada pelanggan baru, jadi perannya perlu dicek lebih lanjut."
    if column == "TotalCharges":
        return f"{label}. Nilainya tumbuh seiring lama berlangganan dan besar tagihan bulanan, jadi informasinya tumpang tindih dengan dua fitur itu."
    if column == "MonthlyCharges":
        return f"{label}. Tagihan yang besar bisa mendorong pelanggan membandingkan dengan operator lain. Ini dugaan yang perlu diuji, bukan hasil model."
    stats = rate_of(m, column, value)
    if stats:
        relation = "di atas" if stats["rate"] > base else "di bawah"
        return f"{label}. Pelanggan dengan kondisi ini churn {pct(stats['rate'])} di data, {relation} rata-rata {pct(base)} ({num(stats['n'])} pelanggan)."
    return label


def importance(m):
    items = m["final"]["importances"]
    top3, top5 = items[:3], items[:5]
    lines = [
        f"Tiga fitur teratas adalah {join_list([short(i['label']) for i in top3])}. "
        f"Ketiganya menyumbang {pct(sum(i['importance'] for i in top3))} dari seluruh kepentingan, "
        f"dan 10 fitur teratas menyumbang {pct(sum(i['importance'] for i in items[:10]))}."
    ]
    lines += [explain_feature(item, m) for item in top3]
    lazim = ["tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService", "PaymentMethod"]
    matched = [i for i in top5 if i["column"] in lazim]
    if len(matched) >= 3:
        lines.append(
            f"{len(matched)} dari 5 fitur teratas ({join_list([short(i['label']) for i in matched])}) memang lazim dikaitkan dengan churn "
            "di industri telekomunikasi, jadi hasilnya sesuai dengan intuisi domain."
        )
    else:
        lines.append(f"Hanya {len(matched)} dari 5 fitur teratas yang lazim dikaitkan dengan churn, jadi hasilnya perlu dicek lebih lanjut dengan data domain.")
    top_groups = grouped_importance(m)[:3]
    lines.append(
        "Fitur kategorikal dipecah menjadi beberapa kolom one-hot, sehingga kepentingannya tersebar. Setelah dijumlahkan per kolom asli, urutan teratas adalah "
        + join_list([f"{short(config.FEATURE_LABELS.get(col, col))} ({pct(score)})" for col, score in top_groups])
        + "."
    )
    lines.append(
        "Kepentingan fitur menunjukkan seberapa sering fitur dipakai pohon untuk memisahkan kelas dan bukan bukti sebab akibat. "
        "Fitur numerik dengan banyak nilai unik cenderung mendapat skor lebih tinggi."
    )
    return lines


def confusion(m):
    c = m["final"]["confusion"]
    pos, neg = c["fn"] + c["tp"], c["tn"] + c["fp"]
    precision = c["tp"] / (c["tp"] + c["fp"])
    lines = [
        f"Dari {num(pos)} pelanggan yang benar-benar churn di data uji, model menangkap {num(c['tp'])} ({pct(c['tp'] / pos)}) dan melewatkan {num(c['fn'])}.",
        f"Dari {num(neg)} pelanggan yang bertahan, {num(c['fp'])} ({pct(c['fp'] / neg)}) salah ditandai akan churn.",
        f"Dari {num(c['tp'] + c['fp'])} pelanggan yang ditandai churn, {num(c['tp'])} ({pct(precision)}) benar-benar pergi.",
    ]
    if c["fn"] < c["fp"]:
        lines.append(
            f"Model lebih sering salah tandai ({num(c['fp'])}) daripada melewatkan churn ({num(c['fn'])}). "
            "Untuk retensi pilihan ini masuk akal, karena kehilangan pelanggan umumnya dianggap lebih mahal daripada memberi promo yang ternyata tidak perlu."
        )
    else:
        lines.append(
            f"Model lebih sering melewatkan churn ({num(c['fn'])}) daripada salah tandai ({num(c['fp'])}), jadi kelas churn masih kurang tertangkap. "
            "Pelanggan yang terlewat pergi tanpa upaya retensi."
        )
    return lines


def roc(m):
    auc = m["final"]["auc"]
    c = m["final"]["confusion"]
    band = "sangat baik" if auc >= 0.9 else "baik" if auc >= 0.8 else "cukup" if auc >= 0.7 else "lemah"
    return [
        f"AUC model {dec(auc)}. Artinya jika satu pelanggan churn dan satu pelanggan bertahan diambil acak, "
        f"model memberi skor risiko lebih tinggi ke pelanggan churn pada sekitar {pct(auc, 0)} pasangan.",
        f"Nilai ini termasuk {band}. Tebakan acak bernilai 0,50 dan model sempurna bernilai 1,00.",
        f"Pada ambang {dec(config.THRESHOLD)}, model berada di titik dengan {pct(c['tp'] / (c['tp'] + c['fn']), 0)} churn tertangkap dan "
        f"{pct(c['fp'] / (c['fp'] + c['tn']), 0)} pelanggan setia salah ditandai. Ambang yang lebih rendah menangkap lebih banyak churn tetapi menambah salah tandai.",
    ]


def report(m):
    f = m["final"]
    churn, stay = f["report"]["churn"], f["report"]["stay"]
    baseline = 1 - m["dataset"]["churn_rate"]
    lines = [
        f"Untuk kelas churn, precision {dec(churn['precision'])} berarti dari 100 pelanggan yang ditandai churn sekitar {round(churn['precision'] * 100)} benar-benar pergi. "
        f"Recall {dec(churn['recall'])} berarti dari 100 pelanggan yang benar-benar pergi sekitar {round(churn['recall'] * 100)} berhasil ditangkap. "
        f"F1-score {dec(churn['f1-score'])} adalah rata-rata harmonik keduanya."
    ]
    if stay["f1-score"] > churn["f1-score"]:
        lines.append(f"Kelas bertahan lebih mudah ditebak (F1 {dec(stay['f1-score'])}) daripada churn (F1 {dec(churn['f1-score'])}) karena jumlahnya jauh lebih banyak.")
    lead = f["test_acc"] - baseline
    if lead < 0.05:
        lines.append(
            f"Akurasi {pct(f['test_acc'])} hanya {pts(lead)} poin di atas tebakan semua bertahan ({pct(baseline)}), "
            "jadi F1 dan recall churn lebih layak dipakai untuk menilai model ini."
        )
    else:
        lines.append(f"Akurasi {pct(f['test_acc'])} berada {pts(lead)} poin di atas tebakan semua bertahan ({pct(baseline)}).")
    macro, weighted = f["report"]["macro"]["f1-score"], f["report"]["weighted"]["f1-score"]
    if weighted > macro:
        lines.append(f"F1 rata-rata makro ({dec(macro)}) lebih rendah daripada rata-rata tertimbang ({dec(weighted)}) karena kelas churn yang lebih sulit dihitung setara dengan kelas bertahan.")
    return lines


def conclusion(m):
    f = m["final"]
    churn = f["report"]["churn"]
    gap = f["train_acc"] - f["test_acc"]
    consistent = abs(f["cv_acc"] - f["test_acc"]) < 0.02 and f["cv_std"] < 0.015
    top_groups = grouped_importance(m)[:3]
    lines = [
        f"Konfigurasi terbaik adalah n_estimators {f['n_estimators']} dan max_depth {depth_text(f['max_depth'])}.",
        f"Pada data uji model mencapai akurasi {pct(f['test_acc'])}, recall churn {pct(churn['recall'])}, F1 churn {dec(churn['f1-score'])}, dan AUC {dec(f['auc'])}.",
        f"Akurasi validasi silang {pct(f['cv_acc'])} (simpangan {pts(f['cv_std'])} poin) dan akurasi uji {pct(f['test_acc'])} berbeda {pts(f['cv_acc'] - f['test_acc'])} poin, "
        + ("jadi hasilnya konsisten pada data yang tidak dipakai untuk memilih parameter." if consistent else "jadi kestabilan model masih perlu diperiksa."),
        f"Selisih akurasi latih dan uji {pts(gap)} poin, "
        + ("masih di bawah batas overfitting." if gap <= config.OVERFIT_GAP else "di atas batas overfitting yang dipakai."),
        f"Faktor yang paling layak dipantau untuk retensi adalah {join_list([short(config.FEATURE_LABELS.get(col, col)).lower() for col, _ in top_groups])}.",
    ]
    return lines


def home(m):
    f = m["final"]
    return (
        f"Random Forest dengan {f['n_estimators']} pohon dan kedalaman maksimum {depth_text(f['max_depth'])} dilatih pada {num(m['dataset']['n_train'])} pelanggan "
        f"dan menangkap {pct(f['report']['churn']['recall'], 0)} pelanggan churn pada data uji."
    )


def explain_prediction(score, factors, m):
    threshold = config.THRESHOLD
    base = m["dataset"]["churn_rate"]
    n_trees = m["final"]["n_estimators"]
    level = "tinggi" if score >= threshold else "sedang" if score >= threshold * 0.6 else "rendah"
    diff = score - threshold

    lines = [
        f"Skor risiko {pct(score, 0)} adalah rata-rata prediksi {n_trees} pohon. Skor di atas {pct(threshold, 0)} ditandai berisiko churn. "
        "Karena model dilatih dengan class_weight balanced, skor ini cocok untuk membandingkan urutan risiko antarpelanggan dan bukan peluang yang terkalibrasi."
    ]
    if abs(diff) < 0.05:
        lines.append("Skor sangat dekat dengan ambang, jadi hasilnya mudah berbalik jika sedikit data pelanggan berbeda.")
    else:
        lines.append(f"Skor berada {pts(diff, 0)} poin {'di atas' if diff > 0 else 'di bawah'} ambang.")

    above = sorted([f for f in factors if f["rate"] > base], key=lambda f: f["rate"], reverse=True)
    if above:
        lines.append(
            "Dari profil ini, kondisi yang di data paling sering berujung churn adalah "
            + join_list([f"{d['label']}: {d['value']} ({pct(d['rate'])})" for d in above[:2]])
            + f", dibanding rata-rata {pct(base)}."
        )
    else:
        lines.append(f"Tidak ada kondisi pada profil ini yang tingkat churn-nya di atas rata-rata data ({pct(base)}).")

    if level == "rendah":
        advice = ["Skor rendah. Pertahankan kualitas layanan dan pantau secara berkala."]
    else:
        advice = [ADVICE[key] for key in ((f["column"], f["raw"]) for f in above) if key in ADVICE][:3]
        if not advice:
            advice = ["Tidak ada faktor tunggal yang dominan, tinjau profil pelanggan ini secara manual."]
    return {"level": level, "paragraphs": lines, "advice": advice}


def trail_text(trail):
    return " dan ".join(f"{'Ya' if yes else 'Tidak'} pada “{question}”" for question, yes in trail)


def leaves(node):
    if node["leaf"]:
        return [node]
    return leaves(node["no"]) + leaves(node["yes"])


def tree_example_text(t):
    threshold = config.THRESHOLD
    lines = [
        f"Ini pohon pertama dari {t['n_trees']} pohon di hutan. Kotak putih adalah pertanyaan, cabang kanan berarti Ya dan cabang kiri berarti Tidak. "
        f"Kotak berwarna adalah ujung jalur yang berisi skor churn milik pohon ini, pink untuk skor di atas {pct(threshold, 0)} dan biru untuk yang di bawahnya.",
        f"Pertanyaan teratasnya adalah “{t['root']['question']}”. Pertanyaan ini dipilih karena paling tajam memisahkan pelanggan churn dari yang bertahan, "
        "dari sebagian fitur yang diundi acak untuk pohon ini.",
    ]
    ends = sorted(leaves(t["root"]), key=lambda node: node["score"])
    low, high = ends[0], ends[-1]
    if low is not high:
        lines.append(
            f"Skor tertinggi pohon ini ({pct(high['score'])}) dimiliki pelanggan dengan jawaban {trail_text(high['trail'])}. "
            f"Skor terendahnya ({pct(low['score'])}) dimiliki pelanggan dengan jawaban {trail_text(low['trail'])}."
        )
    if t["depth"] > t["levels"]:
        lines.append(
            f"Gambar hanya menampilkan {t['levels']} tingkat teratas, sedangkan pohon aslinya sedalam {t['depth']} tingkat. "
            "Angka n adalah jumlah sampel latih yang sampai di kotak itu. Satu pohon bisa keliru, karena itu keputusan akhir adalah rata-rata skor seluruh pohon."
        )
    else:
        lines.append("Pohon ini cukup dangkal sehingga seluruhnya tampil. Satu pohon bisa keliru, karena itu keputusan akhir adalah rata-rata skor seluruh pohon.")
    return lines


def explain_walk(walk, score):
    threshold = config.THRESHOLD
    scores, n, n_churn = walk["scores"], walk["n_trees"], walk["n_churn"]
    low, high = min(scores), max(scores)
    forest = [
        f"Hutan ini punya {n} pohon. Pada pelanggan ini {n_churn} pohon condong ke churn (skor pohon di atas {pct(threshold, 0)}) dan {n - n_churn} pohon condong bertahan. "
        f"Skor risiko {pct(score, 0)} adalah rata-rata skor semua pohon, bukan hitungan suara biasa.",
        f"Skor antarpohon berkisar {pct(low, 0)} sampai {pct(high, 0)}. "
        + (
            "Pohon-pohon terbelah di sekitar ambang, jadi pelanggan ini berada di wilayah abu-abu."
            if low < threshold <= high
            else "Pohon-pohon cenderung sepakat, jadi hasil hutan cukup meyakinkan untuk pelanggan ini."
        ),
    ]
    steps = walk["steps"]
    first, last = steps[0], steps[-1]
    path = [
        f"Pohon yang diperbesar (nomor {walk['picked'] + 1}) dipilih karena skornya paling dekat dengan skor hutan. "
        f"Pohon ini mengajukan {len(steps)} pertanyaan sebelum sampai di daun dan memberi skor {pct(walk['tree_score'], 0)}.",
        f"Pertanyaan pertamanya, “{first['question']}”, dijawab {'Ya' if first['yes'] else 'Tidak'}. "
        f"Pertanyaan terakhirnya, “{last['question']}”, dijawab {'Ya' if last['yes'] else 'Tidak'}.",
        "Pohon lain mengajukan pertanyaan yang berbeda karena tiap pohon dilatih pada sampel dan fitur acak yang berbeda.",
    ]
    return {"forest_lines": forest, "path_lines": path}


ADVICE = {
    ("Contract", "Month-to-month"): "Tawarkan insentif untuk pindah ke kontrak 1 atau 2 tahun.",
    ("TenureGroup", config.TENURE_LABELS[0]): "Beri perhatian ekstra pada masa awal, misalnya program sambutan dan tindak lanjut layanan di bulan pertama.",
    ("PaymentMethod", "Electronic check"): "Dorong pembayaran otomatis lewat transfer bank atau kartu kredit.",
    ("TechSupport", "No"): "Tawarkan paket dukungan teknis.",
    ("OnlineSecurity", "No"): "Tawarkan paket keamanan online.",
    ("InternetService", "Fiber optic"): "Periksa kualitas dan harga layanan fiber untuk pelanggan ini.",
    ("PaperlessBilling", "Yes"): "Pastikan tagihan digital mudah diakses dan dipahami pelanggan.",
}


def build_all(m):
    return {
        "home": home(m),
        "dataset": {
            "class": dataset_class(m),
            "contract": dataset_contract(m),
            "tenure": dataset_tenure(m),
            "missing": dataset_missing(m),
        },
        "experiments": {"a": scenario_a(m), "b": scenario_b(m), "conclusion": conclusion(m)},
        "model": {
            "importance": importance(m),
            "confusion": confusion(m),
            "roc": roc(m),
            "report": report(m),
        },
    }
