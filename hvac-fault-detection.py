import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
import os
#np.random.seed(42)

# Load dataset
path = (r'C:\Users\SIDDHARTH\Downloads\sid\sid\hvac_fault_detection_balanced (1).csv')
if not os.path.exists(path):
    raise FileNotFoundError(f'Expected dataset at {path} but not found.')
df = pd.read_csv(path)

# Fault label distribution
if 'Fault_Label' not in df.columns:
    raise ValueError('Expected column "Fault_Label" in dataset')
counts = df['Fault_Label'].value_counts()
print('Fault distribution:')
print(counts)
plt.figure(figsize=(5,4))
plt.bar(counts.index.astype(str), counts.values)
plt.title('Fault label distribution')
plt.xlabel('Fault_Label')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.show()

print('Shape:', df.shape)
print('\nColumns:')
print(df.columns.tolist())
print('\nInfo:')
df.info()
print('\nMissing values per column:')
print(df.isna().sum())

print(df.describe().T)

#Replace multiple fault types with "Fault" in the same column
df["Fault_Label"] = df["Fault_Label"].replace(
    {"Cooling Failure": "Fault", "Heating Failure": "Fault", "Sensor Error": "Fault"}
)

# Show counts of updated labels
print(df["Fault_Label"].value_counts())

# Prepare data
feature_cols = [c for c in df.columns if c != 'Fault_Label']
X = df[feature_cols].copy()
y = df['Fault_Label'].copy()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_numeric = pipeline.fit_transform(X[numeric_cols])
X_prepped = pd.DataFrame(X_numeric, columns=numeric_cols)
print('Prepared feature shape:', X_prepped.shape)
print(X_prepped.head())

X_train, X_test, y_train, y_test = train_test_split(X_prepped, y, test_size=0.2, stratify=y, random_state=42)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

rf = RandomForestClassifier(max_depth=None, min_samples_split=2,n_estimators=50)
rf.fit(X_train, y_train)
print('Model trained')

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
print('Classification report:')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks([0,1], ['No Fault','Fault'])
plt.yticks([0,1], ['No Fault','Fault'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import joblib
model_path = 'hvac_fault_detection_model.joblib'
joblib.dump(rf, model_path)
print('Saved trained model to', model_path)

# Load the saved model
loaded_model = joblib.load('hvac_fault_detection_model.joblib')

# Get input from the user
input_data = {}
print("Please enter the values for the following features:")
for col in numeric_cols:
    while True:
        try:
            value = float(input(f"{col}: "))
            input_data[col] = [value]
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

# Create a DataFrame from the input data
new_data = pd.DataFrame(input_data)

# Apply the same preprocessing pipeline used for training
# First, apply the imputer and scaler to the numeric columns
new_data_numeric = pipeline.transform(new_data[numeric_cols])
new_data_prepped = pd.DataFrame(new_data_numeric, columns=numeric_cols)

# Predict the fault label
prediction = loaded_model.predict(new_data_prepped)
print('Prediction:', prediction)

# Predict the probability of each class
proba = loaded_model.predict_proba(new_data_prepped)
print('Prediction probabilities:', proba)


# Save trained model & pipeline

model_path = r"C:\Users\SIDDHARTH\Downloads\sid\sid\hvac_fault_detection_model.joblib"
pipeline_path = "pipeline.joblib"

joblib.dump(pipeline, pipeline_path)

print(f"✅ Saved trained model to {model_path}")
print(f"✅ Saved preprocessing pipeline to {pipeline_path}")

# --------------------------
# Load & test on new input
# --------------------------
loaded_model = joblib.load(model_path)
loaded_pipeline = joblib.load(pipeline_path)

# Example test (instead of manual input)
test_input = pd.DataFrame([{
    "Room_Temp_C": 23,
    "Supply_Air_Temp_C": 17,
    "Return_Air_Temp_C": 22,
    "Outdoor_Temp_C": 30,
    "Fan_Status_%": 100,
    "Compressor_Status_%": 60,
    "Cooling_Valve_%": 40,
    "Heating_Valve_%": 0,
    "Power_Usage_kWh": 18,
    "Temp_Setpoint_C": 23
}])

# Preprocess & predict
test_prepped = loaded_pipeline.transform(test_input)
prediction = loaded_model.predict(test_prepped)[0]
proba = loaded_model.predict_proba(test_prepped)[0]

print("\n🔎 Test Prediction:", prediction)
print("🔎 Probabilities:", dict(zip(loaded_model.classes_, proba)))
