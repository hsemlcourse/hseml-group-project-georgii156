from __future__ import annotations

from datetime import date, time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="NYC Taxi Trip Duration",
    page_icon="🚕",
    layout="centered",
)

st.title("🚕 NYC Taxi Trip Duration Prediction")
st.write(
    "Введите параметры поездки, а модель предскажет примерную длительность "
    "поездки в секундах и минутах."
)

with st.form("trip_form"):
    col1, col2 = st.columns(2)

    with col1:
        vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)
        passenger_count = st.number_input(
            "Количество пассажиров",
            min_value=0,
            max_value=9,
            value=1,
            step=1,
        )
        pickup_date = st.date_input("Дата поездки", value=date(2016, 3, 15))
        pickup_time = st.time_input("Время поездки", value=time(14, 30))

    with col2:
        store_and_fwd_flag = st.selectbox("Store and forward flag", ["N", "Y"])
        pickup_latitude = st.number_input("Pickup latitude", value=40.748817, format="%.6f")
        pickup_longitude = st.number_input("Pickup longitude", value=-73.985428, format="%.6f")
        dropoff_latitude = st.number_input("Dropoff latitude", value=40.758896, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff longitude", value=-73.985130, format="%.6f")

    submitted = st.form_submit_button("Предсказать длительность")

if submitted:
    pickup_datetime = f"{pickup_date} {pickup_time.strftime('%H:%M:%S')}"

    payload = {
        "vendor_id": vendor_id,
        "pickup_datetime": pickup_datetime,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "store_and_fwd_flag": store_and_fwd_flag,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            st.success("Предсказание успешно получено")
            st.metric(
                "Длительность поездки",
                f"{result['predicted_trip_duration_minutes']} минут",
            )
            st.write(
                f"Примерно **{result['predicted_trip_duration_seconds']} секунд**."
            )
            with st.expander("Отправленный запрос"):
                st.json(payload)
        else:
            st.error("API вернул ошибку")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Не удалось подключиться к FastAPI. "
            "Сначала запустите API командой: uvicorn app.api:app --reload"
        )
    except requests.exceptions.Timeout:
        st.error("API слишком долго не отвечает. Попробуйте ещё раз.")
