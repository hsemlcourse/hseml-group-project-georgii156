# NYC Taxi Trip Duration Prediction

**Студент:** [Матюхин Георгий Александрович]  
**Группа:** [БИВ238]  
**Курс:** Machine Learning Group Project

## Project Overview
Prediction of taxi trip duration in New York City using Kaggle dataset.

## Dataset
- Источник: [NYC Taxi Trip Duration на Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

## Criteria Coverage
- **Data Processing:** Full EDA, outlier removal, feature engineering (time, distance), visualizations.
- **Modeling:** Baseline (Linear Reg), 4+ models (KNN, RF, GBM, LGBM, XGB), Ensemble method.
- **Reproducibility:** Docker support, fixed random seeds, `requirements.txt`.

## Installation

### Option 1: Local
```bash
pip install -r requirements.txt
jupyter notebook final_solution.ipynb

## Предотвращение data leakage
- Сплит сделан ДО feature engineering
- Очистка выбросов только на train данных
- Все преобразования (scaling, encoding) fit на train, transform на test
- Использован `train_test_split` с random_state=42

## Baseline
- Linear Regression RMSLE: **0.5976**
- Это наш baseline для сравнения с другими моделями

## Выводы

### Почему ансамбль XGB+LGBM лучший:
1. **Градиентный бустинг** лучше работает с табличными данными
2. **Ансамблирование** снижает variance моделей
3. XGBoost и LightGBM дополняют друг друга (разная архитектура)

### Важнейшие фичи:
1. `distance_km` (86.4%) - основная детерминанта времени поездки
2. `hour` (3.1%) - время суток влияет на трафик
3. `dropoff_latitude` (2.7%) - локация высадки

### Trade-offs:
- **Скорость vs Точность**: XGBoost медленнее, но точнее
- **Интерпретируемость**: сложные модели хуже интерпретируются
- **Overfitting**: добавление регуляризации снизило переобучение