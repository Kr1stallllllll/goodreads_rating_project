import pytest
import pickle
import pandas as pd
import gradio as gr
from src.app import predict_rating, demo  # Импортируем из твоего файла app.py

# Тест 1: Проверка работоспособности функции предсказания
def test_predict_function_runs():
    # Проверяем, что функция не падает при корректных данных
    result = predict_rating(300, 1000, 50)
    assert result is not None

# Тест 2: Проверка формата ответа (ожидаем число в строковом формате)
def test_predict_format():
    result = predict_rating(300, 1000, 50)
    # Проверяем, что результат можно конвертировать в float
    try:
        val = float(result)
        assert isinstance(val, float)
    except ValueError:
        pytest.fail("Функция вернула формат, который не является числом")

# Тест 3: Проверка запуска интерфейса Gradio
def test_gradio_app_launch():
    # Проверяем, что интерфейс создан успешно и является объектом Gradio
    assert isinstance(demo, gr.Blocks)