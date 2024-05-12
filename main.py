import os

import pandas as pd
import numpy as np
import streamlit as st
import folium as fl
import src.utils_function as uf
from streamlit_folium import st_folium

from src.utils import prepare_data, train_model, read_model, remove_pkl_model

st.set_page_config(
    page_title="House price predict App",
)

model_path = 'rf_fitted.pkl'

regression_model = st.sidebar.selectbox(
    "Regression model class",
    ("Ridge", "XGB", "Linear"), on_change=remove_pkl_model
)

type_object = st.sidebar.selectbox("Какой тип недвижемости вы ищите?", ("Квартира", "Студия"), index=0)
train = pd.read_csv("data/realty_data.csv")  # Read data

rooms = 0

if type_object != "Студия":
    rooms = st.sidebar.number_input(
        "Какое количество комнат?",
        1, 20, 1,
    )

floor = st.sidebar.number_input(
    "На каком этаже?",
    1, 100, 1,
)

total_square = st.sidebar.number_input(
    "Выберите площадь квартиры/студии (м^2)?",
    1, 3000, 10,
)

# Moscow
latitude = 55.751244
longitude = 37.618423

geo_start = [latitude, longitude]

m = fl.Map(location=geo_start,
           zoom_start=8,
           tiles='OpenStreetMap')

m.add_child(fl.LatLngPopup())

map_geo = st_folium(m, height=600, width=700)

if st.button('Предсказать цену ...'):

    data = (latitude, longitude)
    if map_geo['last_clicked'] is not None:
        data = uf.get_pos(map_geo['last_clicked']['lat'], map_geo['last_clicked']['lng'])

    if not os.path.exists(model_path):
        train_data = prepare_data()
        train_data.to_csv('total_square.csv')
        train_model(train_data, regression_model)

    model = read_model('rf_fitted.pkl')

    df_default = pd.read_csv("default_data_frame.csv")
    numeric_cols = list(df_default.select_dtypes(include=['number']).columns)
    non_numeric_cols = list(df_default.select_dtypes(exclude=['number']).columns)

    for idx, row in df_default.iterrows():
        for col in numeric_cols:
            if col == 'lat':
                df_default.loc[idx, col] = data[0]
            if col == 'lon':
                df_default.loc[idx, col] = data[1]
            if col == 'total_square':
                df_default.loc[idx, col] = total_square
            if col == 'rooms':
                df_default.loc[idx, col] = rooms
            if col == 'floor':
                df_default.loc[idx, col] = floor
            if col == 'is_studio':
                df_default.loc[idx, col] = type_object == "Студия"
            if col == 'floor_category':
                df_default.loc[idx, col] = uf.level_floor(floor)

    preds = model.predict(df_default)[0]

    st.write(f"Your Price Property based on the information provided is: {round(np.exp(preds), 3)} рублей")
