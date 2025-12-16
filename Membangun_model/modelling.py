import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi User
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sabrinayusrina03" 

# Konfigurasi URI
mlflow.set_tracking_uri("https://dagshub.com/Sabrinayusrina03/eksperimen_SML_SabrinaYusrina.mlflow")

# Load data clean
DATA_PATH = 'preprocessing/laptop_clean.csv' 
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Price_euros"])
y = df["Price_euros"]

# Identifikasi kolom
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include="number").columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

mlflow.set_experiment("Laptop Price Prediction")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("R2:", r2)
