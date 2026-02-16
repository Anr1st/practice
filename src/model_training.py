import os
import time
import json
from pathlib import Path

print("AUTOGLUON: ОБУЧЕНИЕ МОДЕЛИ ДЛЯ КЛАССИФИКАЦИИ ЗАМКОВ")

PROJECT_ROOT = Path(__file__).parent.parent

# Пути
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Создаём папки для результатов
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Параметры обучения
MODEL_NAME = "autogluon_castle_classifier"
TIME_LIMIT = 600  # 10 минут на поиск модели (в секундах)
IMAGE_SIZE = (224, 224)  # Должно совпадать с размером из data_preparation.py

print(f"Данные: {DATA_DIR}")
print(f"Модели будут сохранены в: {MODELS_DIR}")
print(f"⏱Лимит времени на обучение: {TIME_LIMIT} сек ({TIME_LIMIT/60:.1f} мин)")

def check_data():
    """Проверяет, что данные подготовлены правильно."""

    print("\nПРОВЕРЯЮ ДАННЫЕ...")

    required_folders = ["train", "val"]
    class_folders = ["class_0_romanesque", "class_1_gothic", "class_2_renaissance"]

    issues = []

    for split in required_folders:
        split_path = DATA_DIR / split
        if not split_path.exists():
            issues.append(f"Папка {split} не найдена: {split_path}")
            continue

        # Проверяем классы внутри каждой split-папки
        for class_folder in class_folders:
            class_path = split_path / class_folder
            if not class_path.exists():
                issues.append(f"Папка класса не найдена: {class_path}")
            else:
                # Считаем изображения
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
                if len(images) == 0:
                    issues.append(f"Нет изображений в: {class_path}")
                else:
                    print(f" {split}/{class_folder}: {len(images)} фото")

    if issues:
        print("\nПРОБЛЕМЫ С ДАННЫМИ:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("Все данные на месте!")
    return True


def train_with_autogluon():
    """Основная функция обучения с AutoGluon."""

    print("\nЗАПУСКАЮ AUTOGLUON...")

    # Импортируем AutoGluon (делаем это здесь, чтобы видеть ошибки импорта)
    try:
        from autogluon.multimodal import MultiModalPredictor
        import pandas as pd
    except ImportError as e:
        print(f"\nОШИБКА ИМПОРТА: {e}")
        print("   Установите AutoGluon: pip install autogluon")
        return None

    # 1. ПОДГОТОВКА ДАННЫХ В ФОРМАТЕ ДЛЯ AUTOGLUON
    print("\nПОДГОТОВКА ДАННЫХ...")

    # Функция для создания таблицы с путями к изображениям
    def prepare_dataframe(split_name):
        data = []
        split_path = DATA_DIR / split_name

        for class_folder in ["class_0_romanesque", "class_1_gothic", "class_2_renaissance"]:
            class_path = split_path / class_folder
            if not class_path.exists():
                continue

            # Простое отображение имени папки в читаемое название
            class_name_map = {
                "class_0_romanesque": "Романский",
                "class_1_gothic": "Готический",
                "class_2_renaissance": "Ренессанс"
            }
            label = class_name_map.get(class_folder, class_folder)

            # Добавляем все изображения
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_path.glob(ext):
                    data.append({
                        "image": str(img_path),
                        "label": label
                    })

        return pd.DataFrame(data)

    # Создаём DataFrames
    train_df = prepare_dataframe("train")
    val_df = prepare_dataframe("val")

    print(f" Обучающих изображений: {len(train_df)}")
    print(f" Валидационных изображений: {len(val_df)}")

    if len(train_df) == 0:
        print("Нет обучающих данных!")
        return None

    # 2. НАСТРОЙКА И ОБУЧЕНИЕ МОДЕЛИ
    print("\nНАСТРАИВАЮ И ОБУЧАЮ МОДЕЛЬ...")

    # Определяем метки классов
    label_column = "label"

    # Инициализируем предсказатель
    predictor = MultiModalPredictor(
        label=label_column,
        path=str(MODELS_DIR / MODEL_NAME),  # Путь для сохранения
        problem_type="multiclass",  # Многоклассовая классификация
        eval_metric="accuracy",     # Метрика для оценки
        verbosity=2,                # Уровень детализации логов
    )

    # Запускаем обучение
    start_time = time.time()

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,  # Данные для валидации во время обучения
        time_limit=TIME_LIMIT,  # Ограничение по времени
        presets="medium_quality",  # Баланс между скоростью и качеством
    )

    training_time = time.time() - start_time
    print(f"Обучение заняло: {training_time:.1f} сек")

    # 3. ОЦЕНКА МОДЕЛИ
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    import numpy as np

    print("\nОЦЕНИВАЮ КАЧЕСТВО МОДЕЛИ...")

    # Если есть тестовые данные - оцениваем на них
    test_df = prepare_dataframe("test") if (DATA_DIR / "test").exists() else None
    eval_dataset_name = "TEST"
    if test_df is None or len(test_df) == 0:
        print("Тестовая выборка не найдена. Используется валидационная выборка для анализа.")
        test_df = val_df
        eval_dataset_name = "VAL"

    if len(test_df) > 0:
        print(f"Анализируем {len(test_df)} изображений из набора {eval_dataset_name}...")

        # 1. Получаем предсказания и вероятности
        y_true = test_df[label_column]
        y_pred = predictor.predict(test_df)
        y_proba = predictor.predict_proba(test_df) # Вероятности для каждого класса

        # 2. Основные метрики (Accuracy, Precision, Recall, F1)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\nМЕТРИКИ ({eval_dataset_name}):")
        print(f"   Accuracy  (Точность): {accuracy:.2%}")
        print(f"   Precision (Точность взвеш.): {precision:.2%}")
        print(f"   Recall    (Полнота взвеш.):  {recall:.2%}")
        print(f"   F1-score  (Гармоническая):   {f1:.2%}")

        # 3. Детальный отчет по классам
        print("\nОтчет по классам:")
        print(classification_report(y_true, y_pred, target_names=predictor.class_labels))

        # 4. Матрица ошибок
        print("\nМатрица ошибок (Что с чем путает модель):")
        # Строки - истина, столбцы - предсказание
        cm = confusion_matrix(y_true, y_pred, labels=predictor.class_labels)
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in predictor.class_labels], 
                                 columns=[f"Pred_{c}" for c in predictor.class_labels])
        print(cm_df)

    # 4. СОХРАНЕНИЕ РЕХУЛЬТАТОВ
        print("\nСОХРАНЯЮ РЕЗУЛЬТАТЫ...")
        
        analysis_df = test_df.copy()
        analysis_df['predicted_label'] = y_pred
        analysis_df['confidence'] = y_proba.max(axis=1) # Уверенность модели
        analysis_df['is_correct'] = analysis_df[label_column] == analysis_df['predicted_label']
        
        # Добавляем вероятности по каждому классу отдельно
        for cls in predictor.class_labels:
            analysis_df[f'prob_{cls}'] = y_proba[cls]

        # Сохраняем в CSV
        analysis_csv_path = REPORTS_DIR / f"prediction_analysis_{eval_dataset_name.lower()}.csv"
        analysis_df.to_csv(analysis_csv_path, index=False)
        print(f"   Файл сохранен: {analysis_csv_path}")
        print("   (Используйте этот файл для поиска ошибок и интерпретации результатов)")

    else:
        print("Нет данных для оценки (ни test, ни val).")
        accuracy = 0.0

    # Модель уже сохранена автоматически в path
    print(f"Модель сохранена в: {MODELS_DIR / MODEL_NAME}")

    # Сохраняем информацию о модели
    model_info = {
        "model_name": MODEL_NAME,
        "training_time_seconds": training_time,
        "time_limit": TIME_LIMIT,
        "image_size": IMAGE_SIZE,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df) if test_df is not None else 0,
        "metrics": {
            "accuracy": accuracy if 'accuracy' in locals() else 0,
            "f1_score": f1 if 'f1' in locals() else 0
        },
        "classes": ["Романский", "Готический", "Ренессанс"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    info_path = REPORTS_DIR / f"{MODEL_NAME}_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"  Информация о модели: {info_path}")

    # 5. ТЕСТИРУЕМ МОДЕЛЬ НА ПРИМЕРАХ
    print("\nТЕСТИРУЮ НА ПРИМЕРАХ...")

    # Берём несколько примеров из валидационной выборки
    if 'analysis_df' in locals() and len(analysis_df) > 0:
        # Показываем 3 самых уверенных правильных ответа
        print(" Топ-3 верных предсказания (высокая уверенность):")
        top_correct = analysis_df[analysis_df['is_correct']].sort_values(by='confidence', ascending=False).head(3)
        for _, row in top_correct.iterrows():
            print(f"   {row[label_column]} -> {row['predicted_label']} ({row['confidence']:.2%}) | {Path(row['image']).name}")

        # Показываем 3 самые грубые ошибки (модель была уверена, но ошиблась)
        print("\n Топ-3 ошибки (модель была уверена, но ошиблась):")
        top_errors = analysis_df[~analysis_df['is_correct']].sort_values(by='confidence', ascending=False).head(3)
        
        if len(top_errors) > 0:
            for _, row in top_errors.iterrows():
                print(f"   Истина: {row[label_column]} -> Предсказала: {row['predicted_label']} ({row['confidence']:.2%}) | {Path(row['image']).name}")
        else:
            print("   Ошибок нет! Идеальная модель.")

    return predictor, model_info

def main():
    """Основная функция запуска обучения."""

    # Проверяем данные
    if not check_data():
        print("\nИсправьте данные и запустите снова.")
        return

    # Запускаем обучение
    result = train_with_autogluon()

    if result is None:
        print("\nОбучение не удалось.")
        return

    predictor, model_info = result

    # Выводим итоги
    print("\nОБУЧЕНИЕ С AUTOGLUON ЗАВЕРШЕНО!")
    
    metrics = model_info.get("metrics", {})
    accuracy = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_score", 0.0)

    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"   Точность (Accuracy): {accuracy:.2%}")
    print(f"   F1-score: {f1:.2%}")
    print(f"   Время обучения: {model_info['training_time_seconds']:.1f} сек")
    print(f"   Обучающих примеров: {model_info['train_samples']}")
    print(f"   Классов: {len(model_info['classes'])}")

    # Оценка качества
    if accuracy >= 0.85:
        print("Отличный результат! Модель готова к использованию.")
    elif accuracy >= 0.75:
        print("Хороший результат. Можно использовать в боте.")
    elif accuracy >= 0.65:
        print("Приемлемый результат. Рассмотрите добавление данных.")
    else:
        print("Результат низкий. Нужно больше данных или аугментации.")
    
    print(f"\nСОЗДАННЫЕ ФАЙЛЫ:")
    print(f"   Модель: {MODELS_DIR / MODEL_NAME}/")
    print(f"   Отчёт: {REPORTS_DIR / f'{MODEL_NAME}_info.json'}")
    csv_report = REPORTS_DIR / f"prediction_analysis_val.csv"
    if csv_report.exists():
         print(f"   CSV для аналитики: {csv_report}")
    else:
         # Проверим test
         csv_report_test = REPORTS_DIR / f"prediction_analysis_test.csv"
         if csv_report_test.exists():
             print(f"   CSV для аналитики: {csv_report_test}")

if __name__ == "__main__":
    main()

