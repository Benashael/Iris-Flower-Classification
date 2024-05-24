import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

st.set_page_config(page_title = "Iris Flower Classification", page_icon = ":flower:")

st.title("Iris Flower Classification :flower:")

iris = load_iris()
X = iris.data
y = iris.target

# Display the dataset
st.header("Iris Dataset")
df = pd.read_csv("IRIS.csv")
st.write(df)

# Train a DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Function to make predictions
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction][0]

# User input features
st.header("User Input Features")

# Options
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

if st.button("Classify"):
    species = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted species is: **{species}**")
