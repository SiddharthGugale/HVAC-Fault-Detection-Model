import streamlit as st
import pandas as pd
import joblib

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="HVAC Fault Detection",
    page_icon="❄️",
    layout="wide",
)

# ------------------- Custom CSS -------------------
st.markdown("""
    <style>
    .main {background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);}
    .stButton>button {
        background: linear-gradient(90deg, #4F8BF9, #1f5ed8);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1f5ed8, #4F8BF9);
        transform: scale(1.05);
    }
    .result-good {
        background: #d4f8e8;
        padding: 1em;
        border-radius: 10px;
        color: #27ae60;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
    }
    .result-bad {
        background: #fde2e1;
        padding: 1em;
        border-radius: 10px;
        color: #c0392b;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1em;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .prob-label {font-weight: bold; font-size: 0.9em;}
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model -------------------
model = joblib.load("hvac_fault_detection_model.joblib")
pipeline = joblib.load("pipeline.joblib")

# ------------------- Title -------------------
st.title("❄️ HVAC Fault Detection System")
st.caption("A smart AI-powered dashboard to monitor HVAC health in real time.")

# ------------------- Sidebar -------------------
st.sidebar.header("⚙️ Input Parameters")
room_temp = st.sidebar.number_input("🌡️ Room Temperature (°C)", value=23.0)
supply_air_temp = st.sidebar.number_input("💨 Supply Air Temp (°C)", value=17.0)
return_air_temp = st.sidebar.number_input("↩️ Return Air Temp (°C)", value=22.0)
outdoor_temp = st.sidebar.number_input("☀️ Outdoor Temp (°C)", value=30.0)
fan_status = st.sidebar.number_input("🌀 Fan Status (%)", value=100.0)
compressor_status = st.sidebar.number_input("⚡ Compressor Status (%)", value=60.0)
cooling_valve = st.sidebar.number_input("❄️ Cooling Valve (%)", value=40.0)
heating_valve = st.sidebar.number_input("🔥 Heating Valve (%)", value=0.0)
power_usage = st.sidebar.number_input("🔋 Power Usage (kWh)", value=18.0)
temp_setpoint = st.sidebar.number_input("🎯 Temp Setpoint (°C)", value=23.0)

# ------------------- Input Summary -------------------
input_df = pd.DataFrame([{
    "Room_Temp_C": room_temp,
    "Supply_Air_Temp_C": supply_air_temp,
    "Return_Air_Temp_C": return_air_temp,
    "Outdoor_Temp_C": outdoor_temp,
    "Fan_Status_%": fan_status,
    "Compressor_Status_%": compressor_status,
    "Cooling_Valve_%": cooling_valve,
    "Heating_Valve_%": heating_valve,
    "Power_Usage_kWh": power_usage,
    "Temp_Setpoint_C": temp_setpoint
}])

with st.expander("📊 Show Input Summary", expanded=False):
    st.dataframe(input_df.style.highlight_max(axis=1, color="lightblue"))

# ------------------- Prediction Section -------------------
if st.button("🔎 Run Prediction", type="primary"):
    try:
        prepped = pipeline.transform(input_df)
        prediction = model.predict(prepped)[0]

        st.subheader("🔮 Prediction Result")
        col1, col2 = st.columns([1, 2])
        
        # Left column: prediction
        with col1:
            if prediction.lower() == "no fault":
                st.markdown(f"<div class='result-good'>✅ {prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-bad'>🚨 {prediction}</div>", unsafe_allow_html=True)

        # Right column: probabilities
        with col2:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(prepped)[0]
                st.markdown("### 📈 Probability Distribution")
                for label, p in zip(model.classes_, proba):
                    st.markdown(f"<span class='prob-label'>{label}:</span> {p:.2%}", unsafe_allow_html=True)
                    st.progress(float(p))
            else:
                st.warning("This model does not support probability estimates.")

        # KPI-style cards
        st.markdown("### ⚡ Key Metrics")
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown("<div class='metric-card'>🌡️<br><b>{} °C</b><br>Room Temp</div>".format(room_temp), unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='metric-card'>💨<br><b>{} °C</b><br>Supply Air</div>".format(supply_air_temp), unsafe_allow_html=True)
        with colC:
            st.markdown("<div class='metric-card'>🔋<br><b>{} kWh</b><br>Power Usage</div>".format(power_usage), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
