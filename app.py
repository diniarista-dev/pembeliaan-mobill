import streamlit as st
import pandas as pd
import joblib

# Load trained models
regression_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")

# Load car catalog
catalog = pd.read_csv("car_catalog.csv")

st.set_page_config(
    page_title="Prediksi Kemampuan Membeli Mobil",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Prediksi Kemampuan Membeli Mobil")
st.write(
    "Aplikasi simulasi prediksi cicilan dan kemampuan membeli mobil."
)

st.sidebar.header("Input Data Pembeli")

monthly_income = st.sidebar.number_input(
    "Penghasilan Bulanan (Rp)",
    min_value=1,
    value=8000000,
    step=500000
)

catalog["label"] = (
    catalog["company"].astype(str)
    + " "
    + catalog["model"].astype(str)
)

selected_car = st.sidebar.selectbox(
    "Pilih Mobil",
    catalog["label"].tolist()
)

car = catalog[catalog["label"] == selected_car].iloc[0]

price_idr = int(car["price_idr"])

st.sidebar.write(f"Harga Mobil: Rp{price_idr:,.0f}")

dp_percent = st.sidebar.slider(
    "Persentase DP (%)",
    10, 80, 30
)

down_payment = int(price_idr * dp_percent / 100)

st.sidebar.write(f"Uang Muka: Rp{down_payment:,.0f}")

loan_tenure = st.sidebar.selectbox(
    "Tenor Cicilan (Bulan)",
    [12, 24, 36, 48, 60]
)

if st.sidebar.button("Prediksi Kemampuan Membeli Mobil"):

    X_reg = pd.DataFrame([{
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    installment = regression_model.predict(X_reg)[0]

    X_clf = pd.DataFrame([{
        "monthly_income": monthly_income,
        "price_idr": price_idr,
        "down_payment": down_payment,
        "loan_tenure": loan_tenure
    }])

    status = classification_model.predict(X_clf)[0]

    dti = installment / monthly_income * 100

    st.subheader("Hasil Prediksi")

    st.metric(
        "Estimasi Cicilan Bulanan",
        f"Rp{installment:,.0f}"
    )

    st.metric(
        "Rasio Cicilan terhadap Gaji",
        f"{dti:.2f}%"
    )

    if status == 1:
        st.success("Status: LAYAK berdasarkan simulasi")
    else:
        st.error("Status: TIDAK LAYAK berdasarkan simulasi")

    st.info(
        "Hasil merupakan simulasi berdasarkan batas DTI 35%, "
        "bukan keputusan resmi bank atau leasing."
    )
