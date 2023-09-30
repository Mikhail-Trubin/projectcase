from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import spacy
import textacy.extract
import subprocess
import json
import os

SetLogLevel(0)

# Проверяем наличие модели
if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

# Устанавливаем Frame Rate
FRAME_RATE = 16000
CHANNELS=1

model = Model("model")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

# предобработкa аудио
mp3 = AudioSegment.from_file('test.mp3')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

# Преобразуем вывод в json
rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)["text"]

# Добавляем пунктуацию
cased = subprocess.check_output('python3 recasepunc/recasepunc.py predict recasepunc/checkpoint', shell=True, text=True, input=text)

# Записываем результат в файл "data.txt"
with open('data.txt', 'w') as f:
    json.dump(cased, f, ensure_ascii=False, indent=4)

nlp = spacy.load('ru_core_news_sm')

# Анализ
doc = nlp(text)

# Извлечение полуструктурированных выражений c ???
statements = textacy.extract.semistructured_statements(doc, "???")

# Вывод результатов

for statement in statements:
    subject, verb, fact = statement
    print(f" - {fact}")