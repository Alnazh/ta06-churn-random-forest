from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
IMG_DIR = BASE_DIR / "static" / "img"

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ESTIMATORS_GRID = [10, 50, 100, 200, 500]
MAX_DEPTH_GRID = [3, 5, 10, 20, None]
SELECTION_TOLERANCE = 0.005
OVERFIT_GAP = 0.10
THRESHOLD = 0.5

TENURE_BINS = [(0, 12), (13, 24), (25, 48), (49, 72)]
TENURE_LABELS = [f"{low}-{high} bulan" for low, high in TENURE_BINS]
INPUT_LIMITS = {"tenure": (0, 72), "MonthlyCharges": (0, 200), "TotalCharges": (0, 15000)}

KEY_FACTORS = ["Contract", "TenureGroup", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]

PALETTE = {
    "ink": "#111111",
    "red": "#ff4d6d",
    "blue": "#2f5bff",
    "yellow": "#ffd93d",
    "grey": "#555555",
}

FEATURE_LABELS = {
    "gender": "Jenis kelamin",
    "SeniorCitizen": "Lansia",
    "Partner": "Punya pasangan",
    "Dependents": "Punya tanggungan",
    "tenure": "Lama berlangganan (bulan)",
    "PhoneService": "Layanan telepon",
    "MultipleLines": "Banyak jalur telepon",
    "InternetService": "Layanan internet",
    "OnlineSecurity": "Keamanan online",
    "OnlineBackup": "Cadangan online",
    "DeviceProtection": "Proteksi perangkat",
    "TechSupport": "Dukungan teknis",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming film",
    "Contract": "Kontrak",
    "PaperlessBilling": "Tagihan tanpa kertas",
    "PaymentMethod": "Metode pembayaran",
    "MonthlyCharges": "Tagihan bulanan",
    "TotalCharges": "Total tagihan",
    "TenureGroup": "Kelompok lama berlangganan",
}

VALUE_LABELS = {
    "Yes": "Ya",
    "No": "Tidak",
    "Female": "Perempuan",
    "Male": "Laki-laki",
    "No internet service": "Tanpa internet",
    "No phone service": "Tanpa telepon",
    "Fiber optic": "Fiber optik",
    "Month-to-month": "Bulanan",
    "One year": "1 tahun",
    "Two year": "2 tahun",
    "Electronic check": "Cek elektronik",
    "Mailed check": "Cek pos",
    "Bank transfer (automatic)": "Transfer bank otomatis",
    "Credit card (automatic)": "Kartu kredit otomatis",
}

FORM_GROUPS = [
    ("Akun", ["tenure", "Contract", "PaymentMethod", "PaperlessBilling", "MonthlyCharges", "TotalCharges"]),
    ("Layanan", ["PhoneService", "MultipleLines", "InternetService"] + INTERNET_ADDONS),
    ("Profil", ["gender", "SeniorCitizen", "Partner", "Dependents"]),
]

PRESETS = {
    "berisiko": {
        "tenure": "2", "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes", "MonthlyCharges": "85.5", "TotalCharges": "",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    },
    "setia": {
        "tenure": "60", "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "No", "MonthlyCharges": "62.4", "TotalCharges": "",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
    },
}
