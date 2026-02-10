import os
import shutil
import random
from PIL import Image
import json

print("ПОДГОТОВКА ДАННЫХ")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к данным
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Настройки разделения (Train/Validation/Test)
SPLIT_RATIO = (0.70, 0.15, 0.15)  # 70% train, 15% val, 15% test
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42

# Стили замков
CLASSES = ['class_0_romanesque', 'class_1_gothic', 'class_2_renaissance']
CLASS_NAMES = ['Романский', 'Готический', 'Ренессанс']

print(f"Исходные данные: {RAW_DATA_DIR}")
print(f"Куда сохранить: {PROCESSED_DATA_DIR}")
print(f"Разделение: Train {SPLIT_RATIO[0]*100}%, Val {SPLIT_RATIO[1]*100}%, Test {SPLIT_RATIO[2]*100}%")
print(f"Размер изображений: {IMAGE_SIZE}")

def prepare_directory_structure():
    """Создаёт чистую структуру папок для обработанных данных."""

    print("\n СОЗДАЮ СТРУКТУРУ ПАПОК...")

    # Удаляем старую папку processed, если она есть
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
        print("  Удалена старая папка 'processed'")

    # Создаём новую структуру
    for split in ['train', 'val', 'test']:
        for class_name in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, class_name), exist_ok=True)

    print("Структура папок создана")
    return True

def check_raw_data():
    """Проверяет исходные данные и собирает статистику."""

    print("\nПРОВЕРЯЮ ИСХОДНЫЕ ДАННЫЕ...")

    stats = {}
    total_images = 0

    for class_dir in CLASSES:
        class_path = os.path.join(RAW_DATA_DIR, class_dir)

        if not os.path.exists(class_path):
            print(f" Папка '{class_dir}' не найдена в {RAW_DATA_DIR}")
            stats[class_dir] = 0
            continue

        # Получаем список изображений
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG')
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(valid_extensions)]

        stats[class_dir] = len(images)
        total_images += len(images)

        # Проверяем первые 3 изображения
        sample_info = []
        for img_name in images[:3]:
            try:
                img_path = os.path.join(class_path, img_name)
                with Image.open(img_path) as img:
                    sample_info.append(f"{img_name} ({img.size[0]}×{img.size[1]})")
            except Exception as e:
                sample_info.append(f"{img_name} (ошибка: {e})")

    print(f"ВСЕГО ИЗОБРАЖЕНИЙ: {total_images}")

    # Проверяем минимальное количество
    min_per_class = min(stats.values()) if stats else 0
    if min_per_class < 30:
        print(f"\n ВНИМАНИЕ: В одном из классов меньше 30 фото!")
        print(f"     Для хорошего качества рекомендуется минимум 50 фото на класс.")

    return stats

def process_and_split_images():
    """
    Обрабатывает и разделяет изображения на train/val/test.
    Возвращает статистику по обработанным файлам.
    """

    print("\nОБРАБАТЫВАЮ И РАЗДЕЛЯЮ ИЗОБРАЖЕНИЯ")

    # Статистика
    processed_stats = {'total': 0, 'failed': 0, 'by_class': {}, 'by_split': {'train': 0, 'val': 0, 'test': 0}}

    for class_idx, class_dir in enumerate(CLASSES):
        class_path = os.path.join(RAW_DATA_DIR, class_dir)

        if not os.path.exists(class_path):
            continue

        # Получаем все изображения
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        all_images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(valid_extensions)]

        if not all_images:
            print(f"В папке '{class_dir}' нет изображений")
            continue

        # Перемешиваем изображения для случайного разделения
        random.seed(RANDOM_SEED + class_idx)  # Разное seed для каждого класса
        random.shuffle(all_images)

        # Вычисляем границы разделения
        n_total = len(all_images)
        n_train = int(n_total * SPLIT_RATIO[0])
        n_val = int(n_total * SPLIT_RATIO[1])
        # n_test = n_total - n_train - n_val

        # Разделяем
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]

        class_display_name = CLASS_NAMES[class_idx]
        print(f"\n {class_display_name} ({class_dir}):")
        print(f"     Всего: {n_total} фото")
        print(f"     Train: {len(train_images)} фото")
        print(f"     Val: {len(val_images)} фото")
        print(f"     Test: {len(test_images)} фото")

        # Инициализируем статистику для класса
        processed_stats['by_class'][class_dir] = {
            'total': n_total, 'train': len(train_images),
            'val': len(val_images), 'test': len(test_images)
        }

        # Обрабатываем и сохраняем изображения для каждого сплита
        for split_name, split_images in [('train', train_images),
                                         ('val', val_images),
                                         ('test', test_images)]:

            for img_name in split_images:
                try:
                    # Открываем и обрабатываем изображение
                    input_path = os.path.join(class_path, img_name)

                    with Image.open(input_path) as img:
                        # Конвертируем в RGB если нужно
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Изменяем размер
                        img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

                        # Сохраняем
                        output_path = os.path.join(PROCESSED_DATA_DIR, split_name, class_dir, img_name)
                        img_resized.save(output_path)

                    # Обновляем статистику
                    processed_stats['total'] += 1
                    processed_stats['by_split'][split_name] += 1

                except Exception as e:
                    processed_stats['failed'] += 1
                    print(f"Ошибка обработки {img_name}: {e}")

        print(f"Обработано: {n_total} фото")

    return processed_stats

def verify_processed_data():
    """Проверяет корректность обработанных данных."""

    print("\n ПРОВЕРЯЮ ОБРАБОТАННЫЕ ДАННЫЕ...")

    issues = []

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(PROCESSED_DATA_DIR, split)

        if not os.path.exists(split_path):
            issues.append(f"Папка {split} не найдена")
            continue

        # Проверяем каждый класс
        for class_dir in CLASSES:
            class_path = os.path.join(split_path, class_dir)

            if not os.path.exists(class_path):
                issues.append(f"Папка {split}/{class_dir} не найдена")
                continue

            # Проверяем наличие изображений
            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not images:
                issues.append(f"Нет изображений в {split}/{class_dir}")
            else:
                # Проверяем размер первого изображения
                try:
                    first_img_path = os.path.join(class_path, images[0])
                    with Image.open(first_img_path) as img:
                        if img.size != IMAGE_SIZE:
                            issues.append(f"Неверный размер в {split}/{class_dir}: {img.size}")
                except:
                    issues.append(f"Ошибка открытия изображения в {split}/{class_dir}")

    if issues:
        print("Найдены проблемы:")
        for issue in issues[:5]:  # Показываем первые 5 проблем
            print(f"     - {issue}")
        if len(issues) > 5:
            print(f"     ... и ещё {len(issues) - 5} проблем")
    else:
        print(" Все данные корректны")

    return len(issues) == 0


def main():
    """Основная функция подготовки данных."""

    print("\n ЗАПУСК ПОДГОТОВКИ ДАННЫХ ДЛЯ AUTOGLUON")

    # 1. Создаём структуру папок
    if not prepare_directory_structure():
        return

    # 2. Проверяем исходные данные
    raw_stats = check_raw_data()

    # Проверяем, есть ли данные для обработки
    total_raw = sum(raw_stats.values())
    if total_raw == 0:
        print("\nНет данных для обработки. Положите изображения в папки:")
        for class_dir in CLASSES:
            print(f"{RAW_DATA_DIR}/{class_dir}/")
        return

    # 3. Обрабатываем и разделяем изображения
    processed_stats = process_and_split_images()

    if processed_stats['total'] == 0:
        print("\nНе удалось обработать ни одного изображения")
        return

    # 5. Проверяем результат
    verify_processed_data()

    # 6. Итоги
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")

    print(f"\n РЕЗУЛЬТАТЫ:")
    print(f"   Обработано изображений: {processed_stats['total']}")
    print(f"   Не удалось обработать: {processed_stats['failed']}")
    print(f"   Train: {processed_stats['by_split']['train']} фото")
    print(f"   Val: {processed_stats['by_split']['val']} фото")
    print(f"   Test: {processed_stats['by_split']['test']} фото")

if __name__ == "__main__":
    main()
