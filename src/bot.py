# src/castle_bot.py
"""
Telegram бот для определения архитектурных стилей замков.
Использует обученную модель AutoGluon и базу знаний styles.json.
"""

import os
import json
import logging
from pathlib import Path
from io import BytesIO

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== КОНФИГУРАЦИЯ ==========
PROJECT_ROOT = Path(__file__).parent.parent

# Пути к модели и базе знаний
MODEL_PATH = PROJECT_ROOT / "models" / "autogluon_castle_classifier"
STYLES_DB_PATH = PROJECT_ROOT / "config" / "styles.json"

# Проверяем наличие файлов
if not MODEL_PATH.exists():
    logger.error(f"❌ Модель не найдена: {MODEL_PATH}")
    logger.error("Сначала обучите модель: python src/train_castle_model.py")
    exit(1)

if not STYLES_DB_PATH.exists():
    logger.error(f"❌ База знаний не найдена: {STYLES_DB_PATH}")
    logger.error("Создайте файл config/styles.json")
    exit(1)

# Загружаем базу знаний о стилях
with open(STYLES_DB_PATH, 'r', encoding='utf-8') as f:
    STYLES_DB = json.load(f)

logger.info(f"✅ Загружена информация о {len(STYLES_DB)} стилях")

# ========== ЗАГРУЗКА МОДЕЛИ ==========
def load_model():
    """Загружает обученную модель AutoGluon."""
    try:
        from autogluon.multimodal import MultiModalPredictor
        logger.info(f"📂 Загрузка модели из {MODEL_PATH}...")
        model = MultiModalPredictor.load(str(MODEL_PATH))
        logger.info("✅ Модель успешно загружена!")
        return model
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Загружаем модель при старте бота
MODEL = load_model()

# ========== КОМАНДЫ БОТА ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🏰 *Добро пожаловать в замковый определитель!*

Я умею определять архитектурный стиль замка по фотографии с точностью 89%!

*Доступные команды:*
/help — подробная инструкция
/styles — список всех стилей
/about — о проекте

*Как пользоваться:* просто отправь мне фото замка!
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📸 *Как правильно фотографировать замки:*

1️⃣ *Ракурс* — фотографируйте фасад или общий вид замка
2️⃣ *Освещение* — лучше днём, чтобы были видны детали
3️⃣ *Качество* — фото не должно быть размытым
4️⃣ *Объекты* — избегайте туристов и деревьев на переднем плане

⚡ *Что я определяю:*
• Романский стиль (XI-XII вв.)
• Готический стиль (XII-XVI вв.)
• Ренессанс (XV-XVII вв.)

❌ *Что я НЕ определяю:*
• Современные постройки
• Интерьеры замков
• Чёрно-белые фото
• Рисунки и чертежи
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def styles_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /styles — список всех стилей"""

    styles_text = "🏛️ *Архитектурные стили замков:*\n\n"

    for style_name, style_info in STYLES_DB.items():
        styles_text += f"• *{style_info['style_name']}*\n"
        styles_text += f"  _{style_info['period']}_\n"
        styles_text += f"  {style_info['description'][:100]}...\n\n"

    styles_text += "\n📎 Отправь фото замка, чтобы узнать его стиль!"

    await update.message.reply_text(styles_text, parse_mode='Markdown')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /about — информация о проекте"""
    about_text = f"""
🧠 *О проекте*

Этот бот создан для классификации архитектурных стилей замков.

📊 *Характеристики модели:*
• Точность: 88.89%
• Классов: 3
• Обучающих фото: 201
• Время инференса: <1 сек

🏰 *Поддерживаемые стили:*
• Романский (XI-XII вв.)
• Готический (XII-XVI вв.)
• Ренессанс (XV-XVII вв.)

🔧 *Технологии:*
• AutoGluon / PyTorch
• Python 3.9
• python-telegram-bot

📅 *Дата релиза:* Февраль 2026
    """
    await update.message.reply_text(about_text, parse_mode='Markdown')


# ========== ОБРАБОТКА ФОТО ==========
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик фотографий — главная функция бота"""

    # Проверяем, загружена ли модель
    if MODEL is None:
        await update.message.reply_text(
            "❌ Модель не загружена. Администратор бота уже решает проблему!"
        )
        return

    # Отправляем статус "печатает..."
    await update.message.chat.send_action(action="typing")

    try:
        # 1. ПОЛУЧАЕМ ФОТО ОТ ПОЛЬЗОВАТЕЛЯ
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # 2. ПОДГОТАВЛИВАЕМ ДАННЫЕ ДЛЯ МОДЕЛИ
        import pandas as pd
        from PIL import Image

        # Сохраняем фото временно
        temp_path = PROJECT_ROOT / "temp" / f"temp_{update.message.from_user.id}.jpg"
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(photo_bytes)

        # Создаём DataFrame для модели
        df = pd.DataFrame({"image": [str(temp_path)]})

        # 3. ДЕЛАЕМ ПРЕДСКАЗАНИЕ
        predictions = MODEL.predict_proba(df)
        label = MODEL.predict(df)[0]

        # Получаем уверенность
        confidence = float(predictions.iloc[0].max())

        # Удаляем временный файл
        os.remove(temp_path)

        # 4. ПОЛУЧАЕМ ИНФОРМАЦИЮ О СТИЛЕ ИЗ БАЗЫ ЗНАНИЙ
        style_info = STYLES_DB.get(label, None)

        if style_info is None:
            await update.message.reply_text(
                f"❌ Стиль '{label}' не найден в базе знаний!"
            )
            return

        # 5. ФОРМИРУЕМ КРАСИВЫЙ ОТВЕТ
        response = format_style_response(label, confidence, style_info)

        # 6. ОТПРАВЛЯЕМ ОТВЕТ
        await update.message.reply_text(response, parse_mode='Markdown')

        # Если уверенность низкая, добавляем рекомендацию
        if confidence < style_info.get('confidence_threshold', 0.6):
            await update.message.reply_text(
                "📸 *Совет:* Уверенность невысокая. Попробуйте:\n"
                "• Сфотографировать замок крупнее\n"
                "• Выбрать более чёткий фасад\n"
                "• Улучшить освещение",
                parse_mode='Markdown'
            )

    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        await update.message.reply_text(
            "❌ Не удалось обработать фото. Попробуйте другое изображение."
        )


def format_style_response(style_name: str, confidence: float, info: dict) -> str:
    """Форматирует красивый ответ с информацией о стиле"""

    confidence_percent = confidence * 100

    # Выбираем эмодзи по уверенности
    if confidence_percent > 85:
        emoji = "🏆"
    elif confidence_percent > 70:
        emoji = "✅"
    elif confidence_percent > 60:
        emoji = "🤔"
    else:
        emoji = "❓"

    # Формируем ответ
    response = f"""
{emoji} *{info['style_name']}*
━━━━━━━━━━━━━━━━━━━━━
📊 *Уверенность:* {confidence_percent:.1f}%

🏛 *Описание:*
{info['description']}

📅 *Период расцвета:*
{info['period']} — {info['period_description']}

🌍 *Регион распространения:*
{info['region']}
{', '.join(info['countries'])}

🔨 *Характерные черты:*
"""
    for feature in info['characteristics'][:6]:  # Показываем первые 6 признаков
        response += f"• {feature}\n"

    response += f"""
🏰 *Известные примеры:*
"""
    for example in info['examples'][:3]:  # Показываем первые 3 примера
        response += f"• {example}\n"

    response += f"""
💡 *Интересный факт:*
{info['fun_fact']}

🎨 *Влияние и происхождение:*
{info['influence']}
"""

    return response


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    await update.message.reply_text(
        "📸 Отправьте мне фотографию замка, чтобы определить его стиль!\n"
        "Используйте /help для получения инструкций."
    )


def main():
    """Главная функция запуска бота"""

    # Токен бота (получите у @BotFather)
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')

    if not TOKEN:
        print("\n" + "=" * 50)
        print("❌ ТОКЕН БОТА НЕ НАЙДЕН!")
        print("=" * 50)
        print("\n1. Перейдите в Telegram к @BotFather")
        print("2. Создайте нового бота: /newbot")
        print("3. Скопируйте полученный токен")
        print("4. Установите токен:")
        print("\n   export TELEGRAM_BOT_TOKEN='ваш_токен_здесь'")
        print("   python src/castle_bot.py\n")
        return

    # Создаём приложение
    app = Application.builder().token(TOKEN).build()

    # Регистрируем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("styles", styles_command))
    app.add_handler(CommandHandler("about", about_command))

    # Регистрируем обработчики сообщений
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Запускаем бота
    print("\n" + "=" * 50)
    print("🤖 ЗАМКОВЫЙ ОПРЕДЕЛИТЕЛЬ ЗАПУЩЕН!")
    print("=" * 50)
    print(f"\n📁 Модель: {MODEL_PATH}")
    print(f"📁 База знаний: {STYLES_DB_PATH}")
    print(f"🎯 Точность модели: 88.89%")
    print(f"\n✅ Бот запущен! Нажмите Ctrl+C для остановки.\n")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
