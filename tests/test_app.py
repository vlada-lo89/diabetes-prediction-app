import pytest
import os
import pandas as pd
from app import load_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_load_data_exists():
    """
    Проверяем, что файл diabetes.csv существует.
    """
    assert os.path.exists("diabetes.csv"), "Файл diabetes.csv не найден!"

def test_load_data():
    """
    Проверяем, что функция load_data загружает данные и возвращает DataFrame.
    """
    data = load_data()
    assert isinstance(data, pd.DataFrame), "Функция load_data должна возвращать DataFrame"
    assert not data.empty, "Датасет не должен быть пустым"
    assert "Outcome" in data.columns, "В датасете должен быть столбец 'Outcome'"

def test_model_training():
    """
    Проверяем, что модель RandomForestClassifier корректно обучается.
    """
    data = load_data()
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)

    assert acc > 0.5, "Точность модели должна быть больше 0.5"
    assert hasattr(model, "estimators_"), "Модель должна быть обучена (отсутствует атрибут estimators_)"

def test_load_data_file_not_found(monkeypatch):
    """
    Проверяем поведение функции load_data при отсутствии файла.
    """
    def mock_read_csv(*args, **kwargs):
        raise FileNotFoundError("Файл не найден")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    with pytest.raises(FileNotFoundError):
        load_data()

