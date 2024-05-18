import csv

# Чтение "плохого" CSV файла и запись исправленного в новый файл
with open("dataset/bad_good_output.csv", "r", encoding="utf-8") as bad_csv_file:
    with open("dataset/good_output.csv", "w", encoding="utf-8", newline="") as corrected_csv_file:
        reader = csv.reader(bad_csv_file)
        writer = csv.writer(corrected_csv_file, delimiter=",", lineterminator="\r")

        for row in reader:
            # Удаление последних двух пустых объектов, если они есть
            corrected_row = row[:-2] if row[-2:] == ["", ""] else row
            writer.writerow(corrected_row)
