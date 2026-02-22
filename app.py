import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

st.set_page_config(page_title="Diabetes Progression Regression")
st.title("Diabetes Progression Prediction using Linear Regression")
# Load Dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Convert to DataFrame (for better display)
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

st.subheader("Dataset Preview")
st.dataframe(df.head())
# Train Model Once
if "model" not in st.session_state:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
# Metrics
mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R² Score): {r2:.2f}")
# Visualization
st.subheader("True vs Predicted Values")

fig, ax = plt.subplots()
ax.scatter(st.session_state.y_test, st.session_state.y_pred)
ax.plot(
    [st.session_state.y_test.min(), st.session_state.y_test.max()],
    [st.session_state.y_test.min(), st.session_state.y_test.max()],
    linestyle="--"
)
ax.set_xlabel("True Values")
ax.set_ylabel("Predicted Values")
ax.set_title("True vs Predicted")

st.pyplot(fig)
# Manual Prediction Section
st.subheader("Predict Diabetes Progression")

user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    prediction = st.session_state.model.predict([user_input])[0]
    st.success(f"Predicted Disease Progression: {prediction:.2f}")
