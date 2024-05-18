import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

# Загрузка данных из отдельных файлов
df_record_name = pd.read_csv('first_bad_inp.csv')
df_record_name_2 = pd.read_csv('second_bad_inp.csv')
df_ref_code = pd.read_csv('good_output.csv')

# Объединение данных в один DataFrame
df_train = pd.DataFrame({
    'record_name': df_record_name['record_name'],
    'record_name_2': df_record_name_2['record_name_2'],
    'ref_code': df_ref_code['ref_code']
})

# Предобработка данных
X = df_train[['record_name', 'record_name_2']]
y = df_train['ref_code']  # Выберите один столбец для целевой переменной

# Преобразование текстовых признаков в числовой формат
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Применение oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_encoded, y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Обучение модели случайного леса
clf = RandomForestClassifier(random_state=42)

# Кросс-валидация модели
cv_scores = cross_val_score(clf, X_resampled, y_resampled, cv=5, scoring='accuracy')

print("Результаты кросс-валидации:")
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.4f}")

# Оценка средней точности модели
mean_accuracy = cv_scores.mean()
print(f"Средняя точность модели: {mean_accuracy:.4f}")

# Обучение и оценка модели на тестовом наборе
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)