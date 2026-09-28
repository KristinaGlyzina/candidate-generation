# Candidate Generation

Решение задачи candidate generation и reranking для поиска релевантных объявлений.

## Подход

Для candidate generation использовалась комбинация sparse- и dense-retrieval:
- word TF-IDF;
- char TF-IDF;
- multilingual E5;
- E5 retrieval по заголовкам;
- location-aware retrieval.

Для reranking использовались retrieval scores и ranks, RRF-признаки, число источников, совпадение локации, семантическая близость и лексическое пересечение между запросом и заголовком.

Финальное ранжирование выполняется с помощью `CatBoostRanker` с `YetiRank`.

## Валидация

Основная локальная метрика — `Macro Recall@50`.

Дополнительно использовались:
- Query Hit@50;
- Pair Hit@50;
- candidate ceiling.

Финальная локальная `Macro Recall@50` — около `0.8043`.

## Структура данных

Ожидаются файлы:

```text
data/
├── benchmark_items.parquet
└── benchmark_queries.parquet
```

Также для воспроизведения финального результата используются подготовленные артефакты моделей и retrieval.

## Запуск

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить финальный pipeline:

```bash
PYTHONPATH=src python -u src/final_pipeline.py
```

После выполнения в корне проекта создаётся:

```text
answer.csv
```

Файл содержит колонки:

```text
query_id,answer
```

где `answer` — до 50 `item_id`, разделённых пробелами.

## Воспроизводимость

Для проверки воспроизводимости использовался SHA256 файла:

```text
outputs/final_v5/top50_long.csv
b3049e047d81794784376b9e18a40d7c09a86c71bc450b49a8b026e5cd2f20a4
```
