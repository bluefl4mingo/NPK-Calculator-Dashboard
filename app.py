import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load a single model
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load all models in the models/ folder
@st.cache_resource
def load_all_models(folder_path):
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]  # Use the file name (without extension) as key
            models[model_name] = load_model(os.path.join(folder_path, filename))
    return models

# Load models
model_folder_path = "models"  # Folder containing the .pkl files
models = load_all_models(model_folder_path)


logocol1, logocol2, logocol3, logocol4, logocol5 = st.columns([0.02, 0.033, 0.03, 0.03, 0.012], vertical_alignment="bottom")
with logocol2:
    st.image(os.path.join("assets", "LogoUGM.svg"), width=106)
with logocol3:
    st.image(os.path.join("assets", "LogoBMKG.svg"), width=90)
with logocol4:
    st.image(os.path.join("assets", "LogoUNS.png"), width=109)

titlecol1, titlecol2, titlecol3 = st.columns([0.1, 0.8, 0.1], vertical_alignment="center")
with titlecol2:
    st.html(f"<h2 style='text-align: center; font-size: 2.5em'>Kalkulator NPK</h2>")
st.divider()

# Input features based on model requirements
st.markdown("<h4 style='text-align: center'>Masukan variabel-variabel input Model di bawah ini untuk menghasilkan kalkulasi NPK</h4>", unsafe_allow_html=True)

# FEATURES LAYOUT
feature_main_container = st.container()
with feature_main_container:
    featcol1, featcol2 = st.columns(2)

# Initialize input data
input_data = {}
    
# KOLOM 1
with featcol1:
    feature1_cont = st.container()
    with feature1_cont:
        feat1img, feat1input, feat1space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat1img:
        st.image(os.path.join("assets", "AirPressure.svg"), width=50)
    with feat1input:
        input_data["Tekanan Udara"] = feat1input.number_input("Tekanan Udara", value=0.0, label_visibility="visible")
    
    feature2_cont = st.container()
    with feature2_cont:
        feat2img, feat2input, feat2space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat2img:
        st.image(os.path.join("assets", "Temperature.svg"), width=50)
    with feat2input:
        input_data["Suhu Avg"] = feat2input.number_input("Suhu Rata-Rata", value=0.0, label_visibility="visible")

    feature3_cont = st.container()
    with feature3_cont:
        feat3img, feat3input, feat3space = st.columns([0.25, 0.75, 0.2], vertical_alignment="bottom")
    with feat3img:
        st.image(os.path.join("assets", "Humidity.svg"), width=50)
    with feat3input:
        input_data["RH"] = feat3input.number_input("Relative Humidity", value=0.0, label_visibility="visible")
    

# Kolom 2 (KANAN)
with featcol2:
    feature5_cont = st.container()
    with feature5_cont:
        feat5space, feat5img, feat5input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat5img:
        st.image(os.path.join("assets", "SolarRadiation.svg"), width=50)
    with feat5input:
        input_data["SR"] = feat5input.number_input("Solar Radiation", value=0.0, label_visibility="visible")
    
    feature6_cont = st.container()
    with feature6_cont:
        feat6space, feat6img, feat6input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat6img:
        st.image(os.path.join("assets", "Rainfall.svg"), width=50)
    with feat6input:
        input_data["Rainfall"] = feat6input.number_input("Curah Hujan", value=0.0, label_visibility="visible")
    
    feature7_cont = st.container()
    with feature7_cont:
        feat7space, feat7img, feat7input = st.columns([0.2, 0.25, 0.75], vertical_alignment="bottom")
    with feat7img:
        st.image(os.path.join("assets", "WindSpeed.svg"), width=50)
    with feat7input:
        input_data["WS"] = feat7input.number_input("Kecepatan Angin", value=0.0, label_visibility="visible")

ftmspace1, ftmimg, ftmtext, ftmspace2 = st.columns([0.4, 0.15, 0.15, 0.4], vertical_alignment="center")

with ftmimg:
    st.image(os.path.join("assets", "ArrowDown.svg"), width=100)

with ftmtext:
    st.html("<h3 style='text-align: left'>Feeding the model</h3>")

# Model selection
model_names = list(models.keys())
mdlspace1, mdlimg, mdlselectbox, mdlspace2 = st.columns([0.3, 0.1, 0.25, 0.2], vertical_alignment="center")

with mdlimg:
    st.image(os.path.join("assets", "MLModel.svg"), width=75)

with mdlselectbox:
    st.html("<h2 style='text-align: left'>NPK Model</h2>")

# Get the selected model
N_model = models["Model N"]
P_model = models["Model P"]
K_model = models["Model K"]

predspace1, predimg, predtext, predspace2 = st.columns([0.4, 0.15, 0.15, 0.4], vertical_alignment="center")

with predimg:
    st.image(os.path.join("assets", "ArrowDown.svg"), width=100)

with predtext:
    st.html("<h3 style='text-align: left'>Calculation</h3>")

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# st.write("Input Data:")
# st.write(input_df)

# Generate predictions using the selected model
if N_model:
    try:
        N_prediction = N_model.predict(input_df)
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.error("No model is selected.")

if P_model:
    try:
        P_prediction = P_model.predict(input_df)
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.error("No model is selected.")

if K_model:
    try:
        K_prediction = K_model.predict(input_df)
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
else:
    st.error("No model is selected.")

ncont, pcont, kcont = st.columns([0.2, 0.2, 0.2], vertical_alignment="center")

with ncont:
    with stylable_container(
                key="custom_container_0",
                css_styles="""
                    {
                        background-color: #639CFF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                        align-items: center;
                        align-content: center;
                        justify-content: center;
                        justify-items: center;
                        padding-right: 5px;
                        padding-left: 5px;
                    }
                    """,
            ):
                st.html(f"<h2 style='text-align: center; font-size: 1.2em', 'text-color= blue'>Nitrogen: {N_prediction[0]:.2f} mg/Kg</h2>")

with pcont:
    with stylable_container(
                key="custom_container_0",
                css_styles="""
                    {
                        background-color: #639CFF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                        align-items: center;
                        align-content: center;
                        justify-content: center;
                        justify-items: center;
                    }
                    """,
            ):
                st.html(f"<h2 style='text-align: center; font-size: 1.2em', 'text-color= blue'>Phosphor: {P_prediction[0]:.2f} mg/Kg</h2>")

with kcont:
    with stylable_container(
                key="custom_container_0",
                css_styles="""
                    {
                        background-color: #639CFF;
                        border: 3px solid #639CFF;
                        border-radius: 10px;
                        align-items: center;
                        align-content: center;
                        justify-content: center;
                        justify-items: center;
                    }
                    """,
            ):
                st.html(f"<h2 style='text-align: center; font-size: 1.2em', 'text-color= blue'>Kalium: {K_prediction[0]:.2f} mg/Kg</h2>")