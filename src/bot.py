import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from autogluon.multimodal import MultiModalPredictor
import pandas as pd

load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

# настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# пути к модели и базе данных
MODEL_PATH = PROJECT_ROOT / "models" / "autogluon_castle_classifier"
STYLES_DB_PATH = PROJECT_ROOT / "config" / "styles.json"

# проверяем наличие файлов
if not MODEL_PATH.exists():
    logger.error("Модель не найдена")
    exit(1)

if not STYLES_DB_PATH.exists():
    logger.error("База данных не найдена")
    exit(1)
with open(STYLES_DB_PATH, 'r', encoding='utf-8') as f:
    STYLES_DB = json.load(f)

def load_model():
    try:
        model = MultiModalPredictor.load(str(MODEL_PATH))
        return model
    except Exception as e:
        logger.error("Ошибка загрузки модели")
        return None

# загружаем модель при старте бота
MODEL = load_model()

# базовые команды бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/start"""
    welcome_text = """
🏰 *Добро пожаловать в архитектурный определитель замков!*

Я умею определять архитектурный стиль по фотографии

Доступные команды:
/help - инструкция
/styles - список стилей, которые определяет бот
/about - о боте

Как пользоваться: просто отправь мне фото
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/help"""
    help_text = """
📸 *Как правильно фотографировать замки:*

1. Ракурс - фотографируй фасад или общий вид замка
2. Освещение - лучше фотографируй днём, чтобы были видны детали
3. Качество - фото не должно быть размытым
4. Объекты - избегай туристов и деревьев на переднем плане
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def styles_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/styles"""

    styles_text = "🏛️ *Архитектурные стили:*\n\n"

    for name, style_info in STYLES_DB.items():
        styles_text += f"*{style_info['name']}*\n"
        styles_text += f"{style_info['period']}\n"

    await update.message.reply_text(styles_text, parse_mode='Markdown')


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/about"""
    about_text = """
Этот бот создан для классификации архитектурных стилей замков.

Характеристики:
- AutoML-модель: AutoGluon
- Классов: 3
- Обучающих фото: 300

Технологии:
- Python
- python-telegram-bot
- Pandas, NumPy, Pillow
    """
    await update.message.reply_text(about_text, parse_mode='Markdown')


# обработка фото пользователя
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if MODEL is None:
        await update.message.reply_text(
            "Ошибка загрузки модели. Дождитесь ответа разработчиков."
        )
        return

    # отправляем статус "печатает..."
    await update.message.chat.send_action(action="typing")

    try:
        # получаем фотку
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # сохраняем фото временно
        temp_path = PROJECT_ROOT / "temp" / f"temp_{update.message.from_user.id}.jpg"
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(photo_bytes)
        df = pd.DataFrame({"image": [str(temp_path)]})

        # получаем ответ модели
        predictions = MODEL.predict_proba(df)
        label = MODEL.predict(df)[0]
        confidence = float(predictions.iloc[0].max())

        # удаляем временный файл
        os.remove(temp_path)

        # получаем информацию из базы данных
        style_info = STYLES_DB.get(label, None)

        if style_info is None:
            await update.message.reply_text(
                f"Стиль '{label}' не найден в базе знаний!"
            )
            return

        # форматирование ответа
        response = format_style_response(label, confidence, style_info)

        # отправляем ответ
        await update.message.reply_text(response, parse_mode='Markdown')

        # если уверенность низкая, добавляем рекомендацию
        if confidence < 0.6:
            await update.message.reply_text(
                "📸 Совет: Уверенность невысокая. Попробуй:\n"
                "- Сфотографировать замок крупнее\n"
                "- Выбрать более чёткий фасад\n"
                "- Улучшить освещение",
                parse_mode='Markdown'
            )

    except Exception as e:
        logger.error("Ошибка обработки фото")
        await update.message.reply_text(
            "Не удалось обработать фото. Попробуй другое изображение."
        )


def format_style_response(style_name: str, confidence: float, info: dict):
    conf = confidence * 100
    # формируем ответ
    response = f"""
*{info['name']}*
Уверенность: {conf:.2f}%

🏛 *Описание:*
{info['description']}

*Период расцвета:*
{info['period']}

*Регион распространения:*
{info['region']}

*Характерные черты:*
"""
    for feature in info['characteristics']:
        response += f"- {feature}\n"

    response += f"""
🏰 *Известные примеры:*
"""
    for example in info['examples']:
        response += f"- {example}\n"

    response += f"""
"""

    return response


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    await update.message.reply_text(
        "📸 Отправь мне фотографию замка, чтобы определить его стиль!\n"
        "Используй /help для получения инструкций."
    )


def main():

    if not TOKEN:
        logger.error("токен не найден")
        return

    app = Application.builder().token(TOKEN).build()

    # регистрируем обработчики команд и сообщений
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("styles", styles_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # запускаем
    print("Бот запущен")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
