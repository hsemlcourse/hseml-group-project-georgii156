# Models

В эту папку нужно положить сохранённые модели:

```text
xgb_model.pkl
lgbm_model.pkl
```

Файлы создаются командой из корня проекта:

```bash
python scripts/train_and_save_models.py
```

Перед запуском скрипта положите `train.csv` в одну из папок:

```text
data/train.csv
notebooks/train.csv
train.csv
```
