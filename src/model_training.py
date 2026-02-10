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
    print("\nОЦЕНИВАЮ КАЧЕСТВО МОДЕЛИ...")

    # Если есть тестовые данные - оцениваем на них
    test_df = prepare_dataframe("test") if (DATA_DIR / "test").exists() else None

    if test_df is not None and len(test_df) > 0:
        print(f"Тестирую на {len(test_df)} изображениях...")

        # Получаем предсказания
        predictions = predictor.predict(test_df)

        # Оцениваем точность
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(test_df[label_column], predictions)
        print(f"Точность на тестовых данных: {accuracy:.2%}")

        # Дополнительная детальная оценка
        evaluation = predictor.evaluate(test_df, metrics=["accuracy", "f1_macro"])
        print(f"Детальная оценка: {evaluation}")
    else:
        print("Тестовых данных нет, оценка только на валидации")
        accuracy = 0.0

    # 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    print("\nСОХРАНЯЮ РЕЗУЛЬТАТЫ...")

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
        "test_accuracy": accuracy if 'accuracy' in locals() else None,
        "classes": ["Романский", "Готический", "Ренессанс"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    info_path = REPORTS_DIR / f"{MODEL_NAME}_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"  📄 Информация о модели: {info_path}")

    # 5. ТЕСТИРУЕМ МОДЕЛЬ НА ПРИМЕРАХ
    print("\nТЕСТИРУЮ НА ПРИМЕРАХ...")

    # Берём несколько примеров из валидационной выборки
    if len(val_df) > 0:
        sample_df = val_df.head(3)  # Первые 3 изображения
        sample_predictions = predictor.predict(sample_df)
        sample_proba = predictor.predict_proba(sample_df)

        print("  Примеры предсказаний:")
        for i, (_, row) in enumerate(sample_df.iterrows()):
            true_label = row[label_column]
            pred_label = sample_predictions[i]
            confidence = max(sample_proba.iloc[i]) * 100

            result = "ok" if true_label == pred_label else "not ok"
            print(f"    {result} Фото {i+1}: Истина={true_label}, Предсказано={pred_label}, Уверенность={confidence:.1f}%")

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
    print("ОБУЧЕНИЕ С AUTOGLUON ЗАВЕРШЕНО!")

    print(f"\nРЕЗУЛЬТАТЫ:")
    if model_info["test_accuracy"] is not None:
        print(f"   Точность на тестовых данных: {model_info['test_accuracy']:.2%}")
    print(f"   Время обучения: {model_info['training_time_seconds']:.1f} сек")
    print(f"   Обучающих примеров: {model_info['train_samples']}")
    print(f"   Классов: {len(model_info['classes'])}")

    # Оценка качества
    if model_info["test_accuracy"] is not None:
        accuracy = model_info["test_accuracy"]
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

if __name__ == "__main__":
    main()
