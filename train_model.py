import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/kos_madiun.csv")
df["ac_num"] = df["ac"].map({"ya": 1, "tidak": 0})
df = df.dropna()

X = df[["jarak_km", "rating", "ac_num"]]
y = df["harga"]

X_latih, X_uji, y_latih, y_uji = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_latih, y_latih)

print("Koefisien :", model.coef_.round(1))
print("Intersep  :", round(model.intercept_, 1))
print("R2 latih  :", round(model.score(X_latih, y_latih), 2))
print("R2 uji    :", round(model.score(X_uji, y_uji), 2))
print("MAE uji   :", round(mean_absolute_error(y_uji, model.predict(X_uji)), 1))

joblib.dump(model, "ml_model/model.pkl")
print("Model tersimpan.")