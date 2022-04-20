import streamlit as st
import pickle
import numpy as np

# Loading the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# To display the title
st.title("Car's Co2 Emission Predictor")


def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://cdn1.sph.harvard.edu/wp-content/uploads/sites/21/2021/12/Vehicle-exhaust_1200x800.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack_url()

engine_size = st.number_input('Engine size')
cylinders = st.selectbox('Cylinders', sorted(df['cylinders'].unique()))
transmission = st.selectbox('Transmission', sorted(df['transmission'].unique()))
fuel_type = st.selectbox('Fuel Type', sorted(df['fuel_type'].unique()))
fuel_consumption_city = st.number_input('Fuel Consumption City')
fuel_consumption_hwy = st.number_input('Fuel Consumption Highway')
fuel_consumption_comb = st.number_input('Combined fuel consumption rating (55% city, 45% highway)')

if st.button("Predict"):
    val = np.array([engine_size, cylinders, transmission, fuel_type, fuel_consumption_city, fuel_consumption_hwy,
                    fuel_consumption_comb])

    val = val.reshape(1, -1)

    predicted = pipe.predict(val)
    predicted = np.round(predicted[0], 2)
    predicted = 'The co2 emission of the car with these features is ' + np.array2string(predicted)
    st.success(predicted)
