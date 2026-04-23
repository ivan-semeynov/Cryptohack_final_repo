# Catalyst Risk Production Model

## Что делает решение

Модель прогнозирует бинарный риск события `target_rise_in_horizon`:
- целевое событие считается равным `1`, если в течение следующих **30 суток** значение `WABT` вырастет минимум на **5°C**;
- входные данные берутся из `unified_3h.csv`;
- обучение повторяет логику исходного Colab-пайплайна, но оформлено как production-ready Python проект.

Алгоритм выбран сознательно: `RandomForestClassifier` + `SimpleImputer(strategy="median")`.
Это хороший баланс между качеством, устойчивостью и простотой эксплуатации.

## Структура проекта

- `model_utils.py` — общая логика подготовки данных, target engineering, split, обучения и инференса.
- `train_model.py` — обучение и сохранение артефактов.
- `predict.py` — пакетный инференс из CSV/JSON.
- `app.py` — FastAPI сервис.
- `requirements.txt` — зависимости.

## 1. Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Обучение модели

```bash
python train_model.py \
  --data /path/to/unified_3h.csv \
  --output-dir ./artifacts
```

После обучения появятся:
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `artifacts/training_summary.json`
- `artifacts/prepared_dataset.csv`
- `artifacts/train_split.csv`
- `artifacts/validation_split.csv`
- `artifacts/test_split.csv`

## 3. Пакетный инференс

### Вариант с CSV

```bash
python predict.py \
  --model-dir ./artifacts \
  --input /path/to/input.csv \
  --output ./predictions.csv
```

### Вариант с JSON

```bash
python predict.py \
  --model-dir ./artifacts \
  --input /path/to/input.json \
  --output ./predictions.json
```

## 4. Запуск API

```bash
MODEL_DIR=./artifacts uvicorn app:app --host 0.0.0.0 --port 8000
```

После этого будут доступны endpoints:
- `GET /health`
- `GET /metadata`
- `POST /predict`

## 5. Формат запроса в API

```json
{
  "records": [
    {
      "dataset_id": 1,
      "диароматика": 10.2,
      "моноароматика": 15.1,
      "полиароматика": 3.4,
      "сера_в_сырье": 0.9,
      "расход_цвсг": 64000,
      "расход_лг": 0,
      "расход_сырья": 127.5,
      "t_отгона_10": 210,
      "t_отгона_50": 280,
      "t_отгона_90": 355,
      "t_отгона_95": 370,
      "плотность_сырья": 0.84,
      "расход_водород": 1200,
      "расход_квенч_всг": 300,
      "t_отгона_100": 390,
      "коксуемость": 0.2,
      "давление_вход": 2.8,
      "расход_серн_бензин": 10,
      "расход_фракц_дт": 20,
      "кратность_цвсг": 500,
      "delta_t_отгона": 180,
      "давление_вход_lag6h": 2.7,
      "давление_вход_lag12h": 2.6,
      "расход_цвсг_lag12h": 63900,
      "расход_сырья_lag6h": 127.1,
      "elapsed_hours": 100,
      "elapsed_share": 0.25
    }
  ]
}
```

Поля можно передавать неполностью: пропуски будут автоматически заполнены медианой, как при обучении.

## 6. Пример вызова API

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "dataset_id": 1,
        "расход_цвсг": 64000,
        "расход_сырья": 127.5,
        "кратность_цвсг": 500,
        "elapsed_hours": 100,
        "elapsed_share": 0.25
      }
    ]
  }'
```

## 7. Что возвращает API

Для каждой записи сервис возвращает:
- `prediction` — 0/1;
- `probability` — вероятность риска;
- `threshold` — рабочий порог, подобранный на validation;
- `risk_level` — low / medium / high;
- `missing_features` — список отсутствующих признаков.

## 8. Производственный процесс

### Шаг 1. Считать исходный CSV
Сервис обучения загружает `unified_3h.csv` и сортирует данные по `dataset_id` и `datetime`.

### Шаг 2. Построить time features
Для каждого `dataset_id` строятся:
- `elapsed_hours` — сколько часов прошло с начала конкретного прогона;
- `elapsed_share` — доля пройденного времени внутри прогона.

### Шаг 3. Сформировать target
Для каждой строки считается максимум `WABT` на горизонте 30 суток вперёд.
Если `max_future_WABT - current_WABT >= 5`, target = 1.

### Шаг 4. Очистить выборку
Удаляются строки, где отсутствуют ключи или все признаки пустые.

### Шаг 5. Сделать temporal split
Для каждого `dataset_id` данные режутся по времени:
- 70% train
- 15% validation
- 15% test

Это важно, потому что случайный shuffle дал бы утечку по времени.

### Шаг 6. Обучить модель
Используется:
- медианная импутация пропусков;
- `RandomForestClassifier`;
- балансировка классов через `sample_weight`.

### Шаг 7. Подобрать threshold
Порог выбирается на validation по максимуму F1.
В проде сервис использует именно этот threshold, а не жёсткий `0.5`.

### Шаг 8. Развернуть API
FastAPI загружает артефакты модели один раз и затем обрабатывает batch-запросы.

## 9. Почему такая реализация подходит для продакшена

- Нет зависимости от ноутбуков и ручного запуска ячеек.
- Отдельный train и отдельный inference.
- Явно сохранены `metadata.json` и threshold.
- API поддерживает batch scoring.
- Пропуски обрабатываются так же, как при обучении.
- Можно завернуть в Docker, Airflow, Kubernetes или любой внутренний MLOps-контур.