import os
import time
import json
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

import torch

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

print("AutoGluon: Обучение модели для классификации замков")

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "autogluon_castle_classifier"
TIME_LIMIT = 1800
IMAGE_SIZE = (224, 224)

print(f"Данные: {DATA_DIR}")
print(f"Модели будут сохранены в: {MODELS_DIR}")
print(f"Лимит времени на обучение: {TIME_LIMIT} сек ({TIME_LIMIT/60:.1f} мин)")

def check_data():
    """Проверяет, что данные подготовлены правильно."""

    print("\nПроверяю данные...")

    required_folders = ["train", "val"]
    class_folders = ["class_0_romanesque", "class_1_gothic", "class_2_renaissance"]

    issues = []

    for split in required_folders:
        split_path = DATA_DIR / split
        if not split_path.exists():
            issues.append(f"Папка {split} не найдена: {split_path}")
            continue

        for class_folder in class_folders:
            class_path = split_path / class_folder
            if not class_path.exists():
                issues.append(f"Папка класса не найдена: {class_path}")
            else:
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
                if len(images) == 0:
                    issues.append(f"Нет изображений в: {class_path}")
                else:
                    print(f" {split}/{class_folder}: {len(images)} фото")

    if issues:
        print("\nПроблемы с данными:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("Все данные на месте!")
    return True


def train_with_autogluon():
    """Основная функция обучения с AutoGluon."""

    print("\nЗапускаю AutoGluon...")

    try:
        from autogluon.multimodal import MultiModalPredictor
        import pandas as pd
    except ImportError as e:
        print(f"\nОшибка импорта: {e}")
        print("   Установите AutoGluon: pip install autogluon")
        return None

    print("\nПодготовка данных...")

    def prepare_dataframe(split_name):
        data = []
        split_path = DATA_DIR / split_name

        for class_folder in ["class_0_romanesque", "class_1_gothic", "class_2_renaissance"]:
            class_path = split_path / class_folder
            if not class_path.exists():
                continue

            class_name_map = {
                "class_0_romanesque": "Романский",
                "class_1_gothic": "Готический",
                "class_2_renaissance": "Ренессанс"
            }
            label = class_name_map.get(class_folder, class_folder)

            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_path.glob(ext):
                    data.append({
                        "image": str(img_path),
                        "label": label
                    })

        return pd.DataFrame(data)

    train_df = prepare_dataframe("train")
    val_df = prepare_dataframe("val")

    print(f" Обучающих изображений: {len(train_df)}")
    print(f" Валидационных изображений: {len(val_df)}")

    if len(train_df) == 0:
        print("Нет обучающих данных!")
        return None

    print("\nНастраиваю и обучаю модель...")

    label_column = "label"

    predictor = MultiModalPredictor(
        label=label_column,
        path=str(MODELS_DIR / MODEL_NAME),
        problem_type="multiclass",
        eval_metric="accuracy",
        verbosity=2,
    )

    start_time = time.time()

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=TIME_LIMIT,
        presets="best_quality",
    )

    training_time = time.time() - start_time
    print(f"Обучение заняло: {training_time:.1f} сек")

    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        precision_recall_fscore_support, f1_score, precision_score, recall_score
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n" + "="*60)
    print("Оценка качества модели")
    print("="*60)

    test_df = prepare_dataframe("test") if (DATA_DIR / "test").exists() else None
    eval_dataset_name = "test"
    if test_df is None or len(test_df) == 0:
        print("Тестовая выборка не найдена. Используется валидационная выборка для анализа.")
        test_df = val_df
        eval_dataset_name = "val"

    if len(test_df) > 0:
        print(f"\nАнализируем {len(test_df)} изображений из набора {eval_dataset_name}...")
        y_true = test_df[label_column]
        y_pred = predictor.predict(test_df)
        y_proba = predictor.predict_proba(test_df)

        accuracy = accuracy_score(y_true, y_pred)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, labels=predictor.class_labels, zero_division=0
        )

        print(f"\n{'='*60}")
        print(f"Основные метрики ({eval_dataset_name})")
        print(f"{'='*60}")
        print(f"   Accuracy (Точность):                    {accuracy:.4f} ({accuracy:.2%})")
        print(f"   Precision (Точность, взвеш.):          {precision_weighted:.4f} ({precision_weighted:.2%})")
        print(f"   Recall (Полнота, взвеш.):              {recall_weighted:.4f} ({recall_weighted:.2%})")
        print(f"   F1-score (Гармоническая, взвеш.):      {f1_weighted:.4f} ({f1_weighted:.2%})")
        print(f"\n   Precision (макросреднее):             {precision_macro:.4f} ({precision_macro:.2%})")
        print(f"   Recall (макросреднее):                 {recall_macro:.4f} ({recall_macro:.2%})")
        print(f"   F1-score (макросреднее):                {f1_macro:.4f} ({f1_macro:.2%})")

        print(f"\n{'='*60}")
        print("Детальные метрики по классам")
        print(f"{'='*60}")
        print(f"{'Класс':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
        print("-" * 60)
        for i, class_label in enumerate(predictor.class_labels):
            print(f"{class_label:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
                  f"{f1_per_class[i]:<12.4f} {support_per_class[i]:<10}")

        print("\nПолный отчет по классам:")
        print(classification_report(y_true, y_pred, target_names=predictor.class_labels, zero_division=0))

        print(f"\n{'='*60}")
        print("Матрица ошибок (Confusion Matrix)")
        print(f"{'='*60}")
        cm = confusion_matrix(y_true, y_pred, labels=predictor.class_labels)
        cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in predictor.class_labels],
                                 columns=[f"Pred: {c}" for c in predictor.class_labels])
        print(cm_df)

        print("\nСоздаю визуализацию confusion matrix...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=predictor.class_labels,
                    yticklabels=predictor.class_labels,
                    cbar_kws={'label': 'Количество образцов'})
        plt.title(f'Confusion Matrix - {eval_dataset_name} Set', fontsize=16, fontweight='bold')
        plt.ylabel('Истинный класс', fontsize=12)
        plt.xlabel('Предсказанный класс', fontsize=12)
        plt.tight_layout()

        cm_plot_path = REPORTS_DIR / f"confusion_matrix_{eval_dataset_name.lower()}.png"
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        print(f"   Сохранено: {cm_plot_path}")
        plt.close()

        print(f"\n{'='*60}")
        print("Сохранение результатов")
        print(f"{'='*60}")

        analysis_df = test_df.copy()
        analysis_df['predicted_label'] = y_pred
        analysis_df['confidence'] = y_proba.max(axis=1)
        analysis_df['is_correct'] = analysis_df[label_column] == analysis_df['predicted_label']

        for cls in predictor.class_labels:
            analysis_df[f'prob_{cls}'] = y_proba[cls]

        analysis_csv_path = REPORTS_DIR / f"prediction_analysis_{eval_dataset_name.lower()}.csv"
        analysis_df.to_csv(analysis_csv_path, index=False)
        print(f"   CSV отчет: {analysis_csv_path}")
        print("   (Используйте этот файл для поиска ошибок и интерпретации результатов)")

        per_class_metrics = {}
        for i, class_label in enumerate(predictor.class_labels):
            per_class_metrics[class_label] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support_per_class[i])
            }

        cm_dict = {}
        for i, true_label in enumerate(predictor.class_labels):
            cm_dict[true_label] = {}
            for j, pred_label in enumerate(predictor.class_labels):
                cm_dict[true_label][pred_label] = int(cm[i, j])

    else:
        print("Нет данных для оценки (ни test, ни val).")
        eval_dataset_name = "N/A"
        accuracy = 0.0
        precision_weighted = 0.0
        recall_weighted = 0.0
        f1_weighted = 0.0
        precision_macro = 0.0
        recall_macro = 0.0
        f1_macro = 0.0
        per_class_metrics = {}
        cm_dict = {}

    print(f"\nМодель сохранена в: {MODELS_DIR / MODEL_NAME}")

    model_info = {
        "model_name": MODEL_NAME,
        "training_time_seconds": training_time,
        "time_limit": TIME_LIMIT,
        "image_size": IMAGE_SIZE,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df) if test_df is not None else 0,
        "evaluation_dataset": eval_dataset_name if 'eval_dataset_name' in locals() else "N/A",
        "metrics": {
            "accuracy": float(accuracy) if 'accuracy' in locals() else 0.0,
            "precision_weighted": float(precision_weighted) if 'precision_weighted' in locals() else 0.0,
            "recall_weighted": float(recall_weighted) if 'recall_weighted' in locals() else 0.0,
            "f1_score_weighted": float(f1_weighted) if 'f1_weighted' in locals() else 0.0,
            "precision_macro": float(precision_macro) if 'precision_macro' in locals() else 0.0,
            "recall_macro": float(recall_macro) if 'recall_macro' in locals() else 0.0,
            "f1_score_macro": float(f1_macro) if 'f1_macro' in locals() else 0.0,
        },
        "per_class_metrics": per_class_metrics if 'per_class_metrics' in locals() else {},
        "confusion_matrix": cm_dict if 'cm_dict' in locals() else {},
        "classes": ["Романский", "Готический", "Ренессанс"],
        "acceptance_criteria": {
            "required_accuracy": 0.80,
            "meets_criteria": float(accuracy) >= 0.80 if 'accuracy' in locals() else False
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    info_path = REPORTS_DIR / f"{MODEL_NAME}_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"   JSON отчет: {info_path}")

    if 'accuracy' in locals():
        print(f"\n{'='*60}")
        print("Проверка критерия приемки")
        print(f"{'='*60}")
        print(f"   Требуемая точность: ≥80%")
        print(f"   Достигнутая точность: {accuracy:.2%}")
        if accuracy >= 0.80:
            print(f"   Критерий приемки выполнен!")
        else:
            print(f"   Критерий приемки не выполнен")
            print(f"   Необходимо улучшить модель (не хватает {0.80 - accuracy:.2%})")

    print("\nТестирую на примерах...")

    if 'analysis_df' in locals() and len(analysis_df) > 0:
        print(" Топ-3 верных предсказания (высокая уверенность):")
        top_correct = analysis_df[analysis_df['is_correct']].sort_values(by='confidence', ascending=False).head(3)
        for _, row in top_correct.iterrows():
            print(f"   {row[label_column]} -> {row['predicted_label']} ({row['confidence']:.2%}) | {Path(row['image']).name}")

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

    if not check_data():
        print("\nИсправьте данные и запустите снова.")
        return

    result = train_with_autogluon()

    if result is None:
        print("\nОбучение не удалось.")
        return

    predictor, model_info = result

    print(f"\n{'='*60}")
    print("Обучение с AutoGluon завершено!")
    print(f"{'='*60}")

    metrics = model_info.get("metrics", {})
    accuracy = metrics.get("accuracy", 0.0)
    f1_weighted = metrics.get("f1_score_weighted", 0.0)
    acceptance = model_info.get("acceptance_criteria", {})
    meets_criteria = acceptance.get("meets_criteria", False)

    print(f"\nИтоговые результаты:")
    print(f"   Точность (Accuracy):              {accuracy:.4f} ({accuracy:.2%})")
    print(f"   F1-score (взвеш.):                {f1_weighted:.4f} ({f1_weighted:.2%})")
    print(f"   Время обучения:                   {model_info['training_time_seconds']:.1f} сек")
    print(f"   Обучающих примеров:               {model_info['train_samples']}")
    print(f"   Валидационных примеров:           {model_info['val_samples']}")
    print(f"   Тестовых примеров:                {model_info['test_samples']}")
    print(f"   Классов:                          {len(model_info['classes'])}")

    print(f"\n{'='*60}")
    print("Критерий приемки")
    print(f"{'='*60}")
    required_accuracy = acceptance.get("required_accuracy", 0.80)
    print(f"   Требуемая точность: ≥{required_accuracy:.0%}")
    print(f"   Достигнутая точность: {accuracy:.2%}")

    if meets_criteria:
        print(f"   Критерий приемки выполнен!")
        print(f"   Модель готова к использованию.")
    else:
        print(f"   Критерий приемки не выполнен")
        print(f"   Необходимо улучшить модель (не хватает {required_accuracy - accuracy:.2%})")
        print(f"   Рекомендации:")
        print(f"      - Увеличить количество обучающих данных")
        print(f"      - Попробовать другие пресеты AutoGluon (best_quality)")
        print(f"      - Увеличить время обучения")
        print(f"      - Проверить качество данных и разметку")

    print(f"\n{'='*60}")
    print("Созданные файлы")
    print(f"{'='*60}")
    print(f"   Модель:                    {MODELS_DIR / MODEL_NAME}/")
    print(f"   JSON отчет:                {REPORTS_DIR / f'{MODEL_NAME}_info.json'}")

    csv_report_val = REPORTS_DIR / "prediction_analysis_val.csv"
    csv_report_test = REPORTS_DIR / "prediction_analysis_test.csv"
    if csv_report_test.exists():
        print(f"   CSV отчет (test):          {csv_report_test}")
    if csv_report_val.exists():
        print(f"   CSV отчет (val):           {csv_report_val}")

    cm_plot_test = REPORTS_DIR / "confusion_matrix_test.png"
    cm_plot_val = REPORTS_DIR / "confusion_matrix_val.png"
    if cm_plot_test.exists():
        print(f"   Confusion Matrix (test):   {cm_plot_test}")
    if cm_plot_val.exists():
        print(f"   Confusion Matrix (val):    {cm_plot_val}")

if __name__ == "__main__":
    main()