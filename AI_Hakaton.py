import csv
import json

from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_distances

# Загрузка модели и токенизатора
model_name = 'DeepPavlov/rubert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# Функция для получения эмбеддинга строки
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


with open("KSR/embeddings.json") as file:
    database = json.load(file)
    database_embeddings = [get_embedding(text) for text in database]

# Обработка новой строки
with open("dataset/first_bad_inp.csv", "w") as input_file:
    with open("dataset/good_output.csv", "w") as output_file:
        input_reader = csv.reader(input_file, delimiter=';')
        output_reader = csv.reader(output_file, delimiter=';')

        good_answer = 0
        all_input = 0

        for input_row, output_row in input_reader, output_reader:
            new_string = input_row
            new_string_embedding = get_embedding(new_string)

            # Нахождение ближайшего соседа
            distances = cosine_distances(new_string_embedding, database_embeddings)
            closest_index = distances.argmin()
            closest_match = database[closest_index]

            # Проверка на неизвестную строку
            threshold = 0.3  # Пример порогового значения

            if distances.min() > threshold:
                k = 5  # количество ближайших соседей для поиска
                closest_indices = distances.argsort()[0][:k]
                closest_matches = [database[i] for i in closest_indices]
                closest_distances = [distances[0][i] for i in closest_indices]

                # Вывод наилучших совпадений
                print("Наилучшие совпадения:")
                for match, dist in zip(closest_matches, closest_distances):
                    print(f"Название: {match}, Расстояние: {dist}")

            else:
                print(f"Лучшее совпадение: {closest_match}")
                if output_row[1] == closest_matches:
                    good_answer += 1
            all_input += 1

        print(f"Программа получила {all_input} запросов. Было дано {good_answer} верных ответов.\n"
              f"Точность программы: {round(good_answer / all_input * 100)}%")

# Взять модель полегче. Сделать эмбединги на 10 строках базы и проверить работособность кода
