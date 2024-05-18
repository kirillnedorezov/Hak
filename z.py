import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

# Загрузка данных
df_train = pd.read_excel('train.xlsx')

# Предобработка данных
X = df_train[['record_name', 'record_name_2']]
y = df_train['ref_code']  # Выберите один столбец для целевой переменной

# Преобразование текстовых признаков в числовой формат
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Применение oversampling
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_encoded, y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Обучение модели случайного леса
clf = RandomForestClassifier()

# Кросс-валидация модели
cv_scores = cross_val_score(clf, X_resampled, y_resampled, cv=5, scoring='accuracy')

print("Результаты кросс-валидации:")
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score}")

# Оценка средней точности модели
mean_accuracy = cv_scores.mean()
print(f"Средняя точность модели: {mean_accuracy}")
