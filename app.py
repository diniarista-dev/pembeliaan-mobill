
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Pengaturan halaman
st.set_page_config(
    page_title="Prediksi Kemampuan Membeli Mobil",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# Memuat model dan katalog
@st.cache_resource
def load_models():
    with open(BASE_DIR / "regression_model.pkl", "rb") as file:
        regression_model = pickle.load(file)

    with open(BASE_DIR / "classification_model.pkl", "rb") as file:
        classification_model = pickle.load(file)

    return regression_model, classification_model


@st.cache_data
def load_catalog():
    return pd.read_csv(BASE_DIR / "car_catalog.csv")


try:
    regression_model, classification_model = load_models()
    catalog = load_catalog()

except Exception as e:
    st.error("Gagal memuat model atau katalog mobil.")
    st.exception(e)
    st.stop()


# Judul aplikasi
st.title("🚗 Prediksi Kemampuan Membeli Mobil")
st.write(
    "Aplikasi ini membantu memperkirakan cicilan mobil "
    "dan menilai kemampuan pengguna berdasarkan penghasilan bulanan."
)

st.divider()

# Input pengguna
st.subheader("Masukkan Data Keuangan")

monthly_income = st.number_input(
    "Penghasilan per Bulan (Rp)",
    min_value=1_000_000,
    value=5_000_000,
    step=500_000
)

st.subheader("Pilih Mobil")

companies = sorted(catalog["company"].dropna().unique())

selected_company = st.selectbox(
    "Merek Mobil",
    companies
)

models = sorted(
    catalog.loc[
        catalog["company"] == selected_company,
        "model"
    ].dropna().unique()
)

selected_model = st.selectbox(
    "Model Mobil",
    models
)

selected_car = catalog[
    (catalog["company"] == selected_company) &
    (catalog["model"] == selected_model)
]

car_price = float(selected_car.iloc[0]["price_idr"])

st.metric(
    "Harga Mobil",
    f"Rp {car_price:,.0f}"
)

dp_percent = st.slider(
    "Persentase Uang Muka (%)",
    min_value=10,
    max_value=80,
    value=30
)

down_payment = car_price * dp_percent / 100

st.write(f"Uang muka: **Rp {down_payment:,.0f}**")

loan_tenure = st.selectbox(
    "Lama Kredit",
    [12, 24, 36, 48, 60],
    index=2,
    format_func=lambda x: f"{x} bulan"
)

st.divider()

# Prediksi
if st.button("Prediksi Kemampuan Membeli Mobil", type="primary"):

    try:
        # Data untuk model regresi
        regression_input = pd.DataFrame({
            "price_idr": [car_price],
            "down_payment": [down_payment],
            "loan_tenure": [loan_tenure]
        })

        predicted_installment = float(
            regression_model.predict(regression_input)[0]
        )

        # Data untuk model klasifikasi
        classification_input = pd.DataFrame({
            "monthly_income": [monthly_income],
            "price_idr": [car_price],
            "down_payment": [down_payment],
            "loan_tenure": [loan_tenure]
        })

        prediction = classification_model.predict(
            classification_input
        )[0]

        dti_ratio = (
            predicted_installment / monthly_income
        ) * 100

        st.subheader("Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediksi Cicilan per Bulan",
                f"Rp {predicted_installment:,.0f}"
            )

        with col2:
            st.metric(
                "Rasio Cicilan terhadap Penghasilan",
                f"{dti_ratio:.2f}%"
            )

        st.divider()

        if prediction == 1:
            st.success(
                "Hasil klasifikasi: Berpotensi mampu membeli mobil."
            )
        else:
            st.warning(
                "Hasil klasifikasi: Belum berpotensi mampu membeli mobil."
            )

        st.caption(
            "Catatan: Hasil merupakan prediksi berdasarkan model "
            "machine learning, bukan keputusan persetujuan kredit "
            "dari lembaga pembiayaan."
        )

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)
