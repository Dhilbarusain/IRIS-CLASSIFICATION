import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KNeighborsClassifier()
model.fit(X_scaled, y)

st.title("🌸 Iris Flower Species Predictor")
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
predicted_class = iris.target_names[prediction][0]

st.write("## 🌼 Predicted Species:")
st.success(f"**{predicted_class}**")

