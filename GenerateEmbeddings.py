import csv
import json
import time

from transformers import DistilBertModel, DistilBertTokenizer
import torch

# Загрузка модели и токенизатора
model_name = 'distilbert-base-multilingual-cased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)


# Функция для получения эмбеддинга строки
def get_embedding(text):
    # Перенос данных на GPU
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    print("dd")
    # Возвращение результата на CPU для дальнейшей работы с NumPy
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Преобразование всех строк в базе данных в эмбеддинги
with open("KSR/classification.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=';')

    # Пропускаем заголовок
    next(reader)

    # Инициализируем словарь
    database = [""] * len(list(reader))

    # Возвращаем указатель файла в начало
    file.seek(0)

    # Пропускаем заголовок еще раз
    next(reader)

    # Читаем строки и формируем словарь
    i = 0
    for row in reader:
        code = row[0]
        name = row[1]
        unit = row[2]
        key = f"{name} ({unit})"

        database[i] = f"{name} {unit}"
        i += 1


database_embeddings = [get_embedding(text) for text in database]
with open("KSR/embeddings.json", "w") as json_file:

    for item in database_embeddings:
        print(item)
        print()
        #json.dump(database_embeddings, json_file, ensure_ascii=False, indent=4)
