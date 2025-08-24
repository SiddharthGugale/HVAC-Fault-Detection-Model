**📌 Problem Statement**

HVAC systems are widely used in residential, commercial, and industrial buildings. However, faults such as sensor failures, leakage, and improper control often go unnoticed, leading to:



1. Higher energy consumption 



2\. Reduced equipment lifespan 



3\. Increased maintenance costs 



**Goal**: Build a machine learning model that can classify whether an HVAC system is operating normally or has a fault.



**Dataset**



The dataset contains HVAC system readings and operational data:



Feature	Description

Timestamp -	      The date and time when the reading was taken

Room\_Temp\_C -	      Actual temperature inside the room (°C)

Supply\_Air\_Temp\_C -   Temperature of air supplied into the room (°C)

Return\_Air\_Temp\_C -   Temperature of air returning from the room (°C)

Outdoor\_Temp\_C -      Outside air temperature (°C)

Fan\_Status\_% -        How much the fan is working (0–100%)

Compressor\_Stat-us\_% -Status of the compressor (0–100%)

Cooling\_Valve\_% -     Cooling valve position (0–100%)

Heating\_Valve\_% -     Heating valve position (0–100%)

Power\_Usage\_kWh -     Energy consumed at that moment

Temp\_Setpoint\_C	-     Target room temperature setpoint (°C)

Fault\_Label -         Target label (e.g., No Fault, or a specific fault type)



**Machine Learning Approach**



1. Preprocessed the dataset (cleaning, encoding, feature scaling).
2. Trained a Random Forest classifier to predict HVAC faults.
3. Selected the best performing model based on accuracy and reliability.
4. Exported the model using joblib and integrated it into a Streamlit app.



**Application Features**



1. Simple, interactive Streamlit web interface.
2. Users can input HVAC system parameters manually.
3. Real-time prediction of system status → No Fault ✅ or Fault ⚠️.
4. Deployed on Streamlit Cloud for public access.



**How to Run Locally**



Clone the repository: git clone https://github.com/SiddharthGugale/HVAC-Fault-Detection-Model.git

cd HVAC-Fault-Detection-Model



Install dependencies: pip install -r requirements.txt



**Run the app:**



streamlit run app.py



🌐 Live Demo



🔗 Try the app here: Streamlit App Link



📌 Future Improvements



**Integrate real-time sensor data.**



Add more advanced ML models (e.g., XGBoost, Gradient Boosting).



Improve UI/UX with additional visualizations and graphs.



**Acknowledgments**



Special thanks to open-source libraries (Streamlit, Scikit-learn, Pandas, NumPy) and dataset contributors.

