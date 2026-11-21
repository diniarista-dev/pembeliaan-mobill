
import streamlit as st
import pandas as pd
import pickle

# Load trained models
with open('regression_model.pkl', 'rb') as file:
    regression_model = pickle.load(file)

with open('classification_model.pkl', 'rb') as file:
    classification_model = pickle.load(file)

# Load car catalog
catalog = pd.read_csv('car_catalog.csv')

# Application title
st.title('Prediksi Kemampuan Membeli Mobil')
st.write('Aplikasi simulasi prediksi cicilan dan kemampuan membeli mobil.')

st.sidebar.header('Input Data Pembeli')

# Input monthly income
monthly_income = st.sidebar.number_input(
    'Penghasilan Bulanan (Rp)',
    min_value=1,
    value=8000000,
    step=500000
)

# Select car
catalog['label'] = (
    catalog['company'] + ' ' + catalog['model']
)

selected_car = st.sidebar.selectbox(
    'Pilih Mobil',
    catalog['label'].tolist()
)

car = catalog[catalog['label'] == selected_car].iloc[0]

price_idr = int(car['price_idr'])

st.sidebar.write(f'Harga Mobil: Rp{price_idr:,.0f}')

# Input DP
dp_percent = st.sidebar.slider(
    'Persentase DP (%)',
    10, 80, 30
)

down_payment = int(price_idr * dp_percent / 100)

st.sidebar.write(
    f'Uang Muka: Rp{down_payment:,.0f}'
)

# Input tenor
loan_tenure = st.sidebar.selectbox(
    'Tenor Cicilan (Bulan)',
    [12, 24, 36, 48, 60]
)

# Prediction button
if st.sidebar.button('Prediksi Kemampuan Membeli Mobil'):

    # Regression input
    X_reg_input = pd.DataFrame([{
        'price_idr': price_idr,
        'down_payment': down_payment,
        'loan_tenure': loan_tenure
    }])

    # Predict monthly installment
    installment = regression_model.predict(
        X_reg_input
    )[0]

    # Classification input
    X_clf_input = pd.DataFrame([{
        'monthly_income': monthly_income,
        'price_idr': price_idr,
        'down_payment': down_payment,
        'loan_tenure': loan_tenure
    }])

    # Predict eligibility
    status = classification_model.predict(
        X_clf_input
    )[0]

    # Calculate DTI
    dti = installment / monthly_income * 100

    st.subheader('Hasil Prediksi')

    st.metric(
        'Estimasi Cicilan Bulanan',
        f'Rp{installment:,.0f}'
    )

    st.metric(
        'Rasio Cicilan terhadap Gaji',
        f'{dti:.2f}%'
    )

    if status == 1:
        st.success('Status: LAYAK berdasarkan simulasi')
    else:
        st.error('Status: TIDAK LAYAK berdasarkan simulasi')

    st.info(
        'Hasil merupakan simulasi berdasarkan model dan '
        'batas DTI 35%, bukan keputusan resmi leasing.'
    )
