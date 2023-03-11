import soundfile as sf
import yaml
import numpy as np
import telegram
from telegram.ext import *
import io
import librosa
from PIL import Image
from inference import (build_classifier_speechbrain, 
    build_label_decoder, build_name_decoder,
    main as recognize)



# Конфигурация
def read_config(filename = './config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def proccess_audio(audio_bytes, target_sr = 16000):
    data, sr = sf.read(io.BytesIO(audio_bytes))
    audio = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    global classifier
    global label_decoder
    global name_mapping

    label_id, score, embedding = recognize(classifier, audio, label_decoder)
    name_id = name_mapping.get(label_id, label_id)

    # Генерируем случайное изображение из numpy и отправляем его в качестве фото-ответа
    img = (embedding/embedding.__abs__().max()) * 255
    img = img.astype(np.uint8)
    img = img[0].transpose()
    img = Image.fromarray(img)

    bio = io.BytesIO()
    bio.name = 'image.png'
    img.save(bio, 'PNG')
    bio.seek(0)

    return bio, name_id, score

# Функция для обработки аудиосообщений
async def audio_message_handler(update: telegram.Update, context: CallbackContext):
    audi_file_id = (update.message.audio or update.message.voice).file_id
    audio_file = await context.bot.get_file(audi_file_id)
    audio_bytes = await audio_file.download_as_bytearray()
    # Отправляем ответное сообщение
    await update.message.reply_text("аудио получено, обрабатываю...")

    img, name_id, score = proccess_audio(audio_bytes)

    await update.message.reply_text(f'Вы похожи на {name_id} на {score} %')
    await context.bot.send_photo(chat_id=update.message.chat_id, photo=img)
    return

async def start_commmand_handler(update, context):
    await update.message.reply_text('Hello! Welcome to this voice2celeb bot!')

# Точка входа
def main(config: dict):
    # Создаем объект Updater и передаем ему токен бота
    application = Application.builder().token(config['api_token']).build()

    # Commands
    application.add_handler(CommandHandler('start', start_commmand_handler))
    application.add_handler(MessageHandler(filters.AUDIO, audio_message_handler))
    application.add_handler(MessageHandler(filters.VOICE, audio_message_handler))

    # Run bot
    application.run_polling(1.0)


if __name__ == '__main__':
    global classifier
    global label_decoder
    global name_mapping
    classifier = build_classifier_speechbrain()
    label_decoder = build_label_decoder()
    name_mapping = build_name_decoder()

    config = read_config(filename = './config.yaml')
    main(config)
