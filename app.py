
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Prediksi Kemampuan Membeli Mobil",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. LOAD MODEL DAN KATALOG MOBIL
# ==========================================

@st.cache_resource
def load_models():
    regression_model = joblib.load(
        BASE_DIR / "regression_model.pkl"
    )

    classification_model = joblib.load(
        BASE_DIR / "classification_model.pkl"
    )

    return regression_model, classification_model


@st.cache_data
def load_catalog():
    return pd.read_csv(BASE_DIR / "car_catalog.csv")


try:
    regression_model, classification_model = load_models()
    catalog = load_catalog()

except Exception as e:
    st.error("Gagal memuat model atau katalog mobil.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. JUDUL APLIKASI
# ==========================================

st.title("🚗 Prediksi Kemampuan Membeli Mobil")

st.write(
    """
    Aplikasi ini digunakan untuk memperkirakan cicilan bulanan
    dan memprediksi kemampuan seseorang dalam membeli mobil
    berdasarkan penghasilan, harga mobil, uang muka, dan tenor.
    """
)

st.info(
    "Hasil kelayakan merupakan simulasi berdasarkan "
    "batas Debt-to-Income Ratio (DTI) sebesar 35%, "
    "bukan keputusan resmi bank atau leasing."
)


# ==========================================
# 4. INPUT DATA PEMBELI
# ==========================================

st.sidebar.header("Data Calon Pembeli")

monthly_income = st.sidebar.number_input(
    "Penghasilan Bulanan (Rp)",
    min_value=1,
    max_value=1_000_000_000,
    value=8_000_000,
    step=500_000
)


# ==========================================
# 5. PILIHAN MOBIL
# ==========================================

st.sidebar.header("Pilihan Mobil")

required_columns = [
    "company",
    "model",
    "body_style",
    "price_idr"
]

if not all(col in catalog.columns for col in required_columns):
    st.error("Kolom katalog mobil tidak sesuai.")
    st.stop()

catalog = catalog.dropna(
    subset=["company", "model", "price_idr"]
).copy()

catalog["label"] = (
    catalog["company"].astype(str)
    + " - "
    + catalog["model"].astype(str)
)

selected_car = st.sidebar.selectbox(
    "Pilih Mobil",
    catalog["label"].tolist()
)

car = catalog[
    catalog["label"] == selected_car
].iloc[0]

price_idr = int(car["price_idr"])

st.sidebar.write(
    f"Harga Mobil: Rp{price_idr:,.0f}"
)

st.sidebar.write(
    f"Jenis Bodi: {car['body_style']}"
)


# ==========================================
# 6. INPUT UANG MUKA (DP)
# ==========================================

dp_percent = st.sidebar.slider(
    "Persentase Uang Muka (DP)",
    min_value=10,
    max_value=80,
    value=30,
    step=1
)

down_payment = int(
    price_idr * dp_percent / 100
)

st.sidebar.write(
    f"Nominal DP: Rp{down_payment:,.0f}"
)


# ==========================================
# 7. INPUT TENOR
# ==========================================

loan_tenure = st.sidebar.selectbox(
    "Tenor Cicilan",
    [12, 24, 36, 48, 60],
    index=2,
    format_func=lambda x: f"{x} Bulan"
)


# ==========================================
# 8. PROSES PREDIKSI
# ==========================================

if st.sidebar.button(
    "Prediksi Kemampuan Membeli Mobil",
    use_container_width=True
):

    # Input regresi
    X_reg_input = pd.DataFrame([{
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    # Prediksi cicilan bulanan
    installment = float(
        regression_model.predict(X_reg_input)[0]
    )

    # Input klasifikasi
    X_clf_input = pd.DataFrame([{
        "monthly_income": monthly_income,
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    # Prediksi kelayakan
    status = int(
        classification_model.predict(X_clf_input)[0]
    )

    # Menghitung DTI
    dti = (
        installment / monthly_income
    ) * 100


    # ======================================
    # 9. MENAMPILKAN HASIL
    # ======================================

    st.divider()

    st.subheader("📊 Hasil Prediksi")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Harga Mobil",
        f"Rp{price_idr:,.0f}"
    )

    col2.metric(
        "Uang Muka",
        f"Rp{down_payment:,.0f}"
    )

    col3.metric(
        "Tenor",
        f"{loan_tenure} Bulan"
    )

    st.divider()

    col4, col5 = st.columns(2)

    col4.metric(
        "Estimasi Cicilan Bulanan",
        f"Rp{installment:,.0f}"
    )

    col5.metric(
        "Rasio Cicilan terhadap Gaji",
        f"{dti:.2f}%"
    )

    st.divider()

    st.subheader("Hasil Klasifikasi")

    if status == 1:
        st.success(
            "LAYAK berdasarkan prediksi model."
        )
    else:
        st.error(
            "TIDAK LAYAK berdasarkan prediksi model."
        )

    st.subheader("Analisis Kemampuan Membayar")

    if dti <= 35:
        st.success(
            f"DTI sebesar {dti:.2f}% berada "
            "pada atau di bawah batas simulasi 35%."
        )
    else:
        st.warning(
            f"DTI sebesar {dti:.2f}% melebihi "
            "batas simulasi 35%."
        )

    st.caption(
        "Hasil merupakan simulasi berdasarkan model "
        "dan asumsi DTI 35%, bukan keputusan resmi leasing. "
        "Prediksi regresi dan klasifikasi merupakan dua "
        "model terpisah sehingga hasilnya dapat berbeda."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Prediksi Kemampuan Membeli Mobil | "
    "Machine Learning dengan Python dan Streamlit"
)
