import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data():
    data = pd.read_csv("diabetes.csv")
    return data

def main():
    st.title("Приложение для предсказания диабета")
    st.write("""Это интерактивное веб-приложение, использующее модель машинного обучения (Random Forest) для классификации наличия диабета на основе медицинских показателей.""")

    st.sidebar.title("Параметры пациента")
    st.sidebar.markdown("Укажите необходимые характеристики ниже:")
    pregnancies = st.sidebar.number_input("Число беременностей (Pregnancies)", min_value=0, max_value=20, value=1, step=1)
    glucose = st.sidebar.slider("Концентрация глюкозы (Glucose)", 0, 200, 100, 1)
    blood_pressure = st.sidebar.slider("Давление (BloodPressure)", 0, 130, 70, 1)
    skin_thickness = st.sidebar.slider("Толщина кожной складки (SkinThickness)", 0, 100, 20, 1)
    insulin = st.sidebar.slider("Уровень инсулина (Insulin)", 0, 900, 30, 1)
    bmi = st.sidebar.number_input("ИМТ (BMI)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
    dpf = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.sidebar.number_input("Возраст (Age)", min_value=1, max_value=120, value=30, step=1)

    show_metrics = st.sidebar.checkbox("Показать метрики качества")

    data = load_data()

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    if st.button("Сделать предсказание"):
        input_data = np.array([
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]).reshape(1, -1)

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error(f"Вероятно, у пациента ДИАБЕТ. (уверенность: {prediction_proba[0][1]:.2f})")
        else:
            st.success(f"Вероятно, у пациента НЕТ диабета. (уверенность: {prediction_proba[0][0]:.2f})")

    if show_metrics:
        st.subheader("Модель: RandomForestClassifier")
        st.write(f"Точность на тесте: {test_acc:.2f}")
        cm = confusion_matrix(y_test, y_pred_test)
        st.write("Confusion Matrix:")
        st.write(cm)

    st.write("---")
    st.write("Ниже - несколько строк исходного датасета:")
    st.dataframe(data.head(10))

if __name__ == "__main__":
    main()