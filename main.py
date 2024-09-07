import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os

# Путь к данным и файл для прогнозирования
data_path = 'dataset'
files = os.listdir(data_path)
prediction_file = '2023-04-02.csv'

# Создаем пустую модель для инкрементального обучения
bst_models = {}

# Словарь для хранения метрик
metrics = []

# Колонки, которые мы будем использовать
selected_columns = ['date', 'model', 'capacity_bytes', 'smart_1_normalized', 'smart_1_raw', 'failure',
                    'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                    'smart_191_raw', 'smart_192_raw', 'smart_193_raw', 'smart_197_raw', 'smart_198_raw']

# Функция для логирования сообщений
def log_message(message):
    print(message)
    with open('log.txt', 'a') as f:
        f.write(message + '\n')

# Функция для проверки распределения классов
def check_class_distribution(y):
    distribution = y.value_counts(normalize=True)
    log_message(f"Распределение классов:\n{distribution}")

# Инициализация импьютеров вне цикла для последовательного использования
imputer_numeric = SimpleImputer(strategy='mean')
imputer_categorical = SimpleImputer(strategy='most_frequent')

# Флаг для определения необходимости подгонки импьютеров
imputers_fitted = False

# Периоды прогноза в месяцах
months_to_predict = [3, 6, 9, 12]

# Обработка каждого файла
for file in files:
    log_message(f"Обрабатываю файл: {file}")
    df = pd.read_csv(os.path.join(data_path, file))

    # Получаем дату из названия файла, если колонки 'date' нет в файле
    if 'date' not in df.columns:
        file_date = pd.to_datetime(file.split('.')[0], format='%Y-%m-%d')
        df['date'] = file_date

    # Оставляем только выбранные колонки, которые есть в текущем файле
    df = df[[col for col in selected_columns if col in df.columns]]

    log_message(f"Размер данных после фильтрации по выбранным колонкам: {df.shape}")

    # Удаляем столбцы, которые содержат только пропущенные значения, кроме 'date'
    cols_to_drop = df.columns[df.isna().all()].difference(['date'])
    df = df.drop(columns=cols_to_drop)

    log_message(f"Размер данных после удаления колонок с пропущенными значениями: {df.shape}")

    # Предобработка данных: заполнение пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns

    if not imputers_fitted:
        # Подгонка импьютеров на первом файле
        imputer_numeric.fit(df[numeric_cols])
        imputer_categorical.fit(df[categorical_cols])
        imputers_fitted = True

    # Заполняем пропущенные значения
    if len(numeric_cols) > 0:
        df[numeric_cols] = imputer_numeric.transform(df[numeric_cols])
    if len(categorical_cols) > 0:
        df[categorical_cols] = imputer_categorical.transform(df[categorical_cols])
    
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64') // 10**9  # Преобразование в UNIX-время

    # Сохраняем 'date' в переменной
    date_col = df['date']
    date_col = pd.to_datetime(date_col, unit='s')
    # Преобразование категориальных переменных в дамми-переменные
    if not categorical_cols.empty:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
 
    # Добавляем прогнозируемые даты для каждого периода
    for month in months_to_predict:
        df[f'date_{month}_months'] = date_col + pd.DateOffset(months=month)

    # Убираем целевую переменную если она есть
    if 'failure' in df.columns:
        # Преобразование datetime колонок в числовой формат
        X = df.drop(columns=['failure'])
        for col in X.select_dtypes(include=['datetime64']).columns:
            X[col] = X[col].astype('int64') // 10**9  # Преобразуем в секунды

        y = df['failure']
        check_class_distribution(y)  # Проверяем распределение классов

        # Проверяем, есть ли больше одного класса в целевой переменной
        if len(y.unique()) > 1:
            # Применение SMOTE только если в классе "failure=1" достаточно примеров
            if y.value_counts().min() >= 2:  # Минимум два примера для SMOTE
                smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
                X_res, y_res = smote.fit_resample(X, y)
            else:
                log_message(f"Пропускаем SMOTE, так как слишком мало примеров для failure=1.")
                X_res, y_res = X, y
        else:
            log_message(f"Пропускаем SMOTE, так как целевая переменная содержит только один класс.")
            X_res, y_res = X, y

        for month in months_to_predict:
            log_message(f"Прогноз на {month} месяцев.")
            
            # Приведение данных к тем же признакам, что и обучающая модель
            if month in bst_models:
                model_features = bst_models[month].feature_names

                # Добавляем недостающие признаки с -1
                for feature in model_features:
                    if feature not in X_res.columns:
                        X_res[feature] = -1

                # Убираем лишние признаки, которых нет в обучающей модели
                X_res = X_res[model_features]

            # Создание DMatrix для текущего файла
            dtrain = xgb.DMatrix(X_res, label=y_res)

            if month not in bst_models:
                log_message(f'Обучаем модель для прогноза на {month} месяцев с нуля.')

                # Проверка на деление на ноль при расчете scale_pos_weight
                if len(y_res[y_res == 1]) > 0:
                    scale_pos_weight = len(y_res[y_res == 0]) / len(y_res[y_res == 1])
                else:
                    scale_pos_weight = 1  # Если класс "1" отсутствует, устанавливаем значение по умолчанию

                # Условие для модели на 12 месяцев
                if month == 12:
                    log_message("Применяем новые параметры для модели на 12 месяцев.")
                    bst_models[month] = xgb.train(params={
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'random_state': 65,
                        'scale_pos_weight': scale_pos_weight,
                        'max_depth': 8,  # Новое значение
                        'learning_rate': 0.05,  # Новое значение
                        'subsample': 0.9,  # Новое значение
                        'colsample_bytree': 0.7,  # Новое значение
                        'min_child_weight': 3,  # Новое значение
                        'gamma': 0.2  # Новое значение
                    }, dtrain=dtrain, num_boost_round=150)  # Новое значение для количества итераций
                else:
                    bst_models[month] = xgb.train(params={
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'random_state': 65,
                        'scale_pos_weight': scale_pos_weight,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }, dtrain=dtrain, num_boost_round=100)
            else:
                log_message(f'Дообучаем модель для прогноза на {month} месяцев.')

                # Проверка на деление на ноль при расчете scale_pos_weight
                if len(y_res[y_res == 1]) > 0:
                    scale_pos_weight = len(y_res[y_res == 0]) / len(y_res[y_res == 1])
                else:
                    scale_pos_weight = 1  # Если класс "1" отсутствует, устанавливаем значение по умолчанию

                # Условие для модели на 12 месяцев
                if month == 12:
                    log_message("Применяем новые параметры для дообучения модели на 12 месяцев.")
                    bst_models[month] = xgb.train(params={
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'random_state': 65,
                        'scale_pos_weight': scale_pos_weight,
                        'max_depth': 8,  # Новое значение
                        'learning_rate': 0.05,  # Новое значение
                        'subsample': 0.9,  # Новое значение
                        'colsample_bytree': 0.7,  # Новое значение
                        'min_child_weight': 3,  # Новое значение
                        'gamma': 0.2  # Новое значение
                    }, dtrain=dtrain, num_boost_round=150, xgb_model=bst_models[month])  # Новое значение для количества итераций
                else:
                    bst_models[month] = xgb.train(params={
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'random_state': 65,
                        'scale_pos_weight': scale_pos_weight,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }, dtrain=dtrain, num_boost_round=100, xgb_model=bst_models[month])

            # Прогнозирование на обучающем наборе данных
            predictions = bst_models[month].predict(dtrain)
            threshold = 0.5
            y_pred = np.round(predictions > threshold)

            # Вычисление метрик
            accuracy = accuracy_score(y_res, y_pred)
            precision = precision_score(y_res, y_pred, zero_division=1)
            recall = recall_score(y_res, y_pred, zero_division=1)
            f1 = f1_score(y_res, y_pred, zero_division=1)

            metrics.append({
                'file': file,
                'month': month,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

            log_message(f"Метрики для файла {file}, прогноз на {month} месяцев: accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1}")
# Читаем файл
log_message(f"Прогноз для файла {prediction_file}")
df_pred = pd.read_csv(os.path.join(prediction_file))

# Получаем дату из названия файла, если колонки 'date' нет в файле
if 'date' not in df_pred.columns:
    file_date = pd.to_datetime(prediction_file.split('.')[0], format='%Y-%m-%d')
    df_pred['date'] = file_date
df_results = df_pred[['serial_number']]
# Оставляем только выбранные колонки, которые есть в текущем файле
df_pred = df_pred[[col for col in selected_columns if col in df_pred.columns]]

# Удаляем столбцы, которые содержат только пропущенные значения, кроме 'date'
cols_to_drop = df_pred.columns[df_pred.isna().all()].difference(['date'])
df_pred = df_pred.drop(columns=cols_to_drop)

# Заполнение пропусков
numeric_cols_pred = df_pred.select_dtypes(include=[np.number]).columns
categorical_cols_pred = df_pred.select_dtypes(include=[object]).columns

# Заполняем пропущенные значения
if len(numeric_cols_pred) > 0:
    df_pred[numeric_cols_pred] = imputer_numeric.transform(df_pred[numeric_cols_pred])
if len(categorical_cols_pred) > 0:
    df_pred[categorical_cols_pred] = imputer_categorical.transform(df_pred[categorical_cols_pred])

# Преобразуем дату в UNIX-время
df_pred['date'] = pd.to_datetime(df_pred['date'])
df_pred['date'] = df_pred['date'].astype('int64') // 10**9

# Преобразование категориальных переменных в дамми-переменные
if not categorical_cols_pred.empty:
    df_pred = pd.get_dummies(df_pred, columns=categorical_cols_pred, drop_first=True)

# Итоговый DataFrame для сохранения результатов
  # Добавляем серийный номер для идентификации

# Прогнозирование для каждого периода
for month in months_to_predict:
    log_message(f"Прогноз для {month} месяцев")

    # Приведение данных к тем же признакам, что и обучающая модель
    model_features = bst_models[month].feature_names

    # Добавляем недостающие признаки с -1
    for feature in model_features:
        if feature not in df_pred.columns:
            df_pred[feature] = -1

    # Убираем лишние признаки
    df_pred_month = df_pred[model_features]

    # Создание DMatrix для файла
    dpred = xgb.DMatrix(df_pred_month)

    # Прогнозирование
    predictions = bst_models[month].predict(dpred)
    
    # Применение порога
    threshold = 0.5
    y_pred = np.round(predictions > threshold).astype(int)  # Преобразуем прогноз в 0 или 1

    # Добавляем результаты прогноза в итоговый DataFrame
    df_results[f'failure_in_{month}_months'] = y_pred

# Сохранение результатов в CSV
output_file = f"predictions_{os.path.basename(prediction_file)}"
df_results.to_csv(output_file, index=False)

log_message(f"Прогнозы сохранены в файл: {output_file}")
# Сохранение обученной модели для каждого месяца
for month in months_to_predict:
    if month in bst_models:
        bst_models[month].save_model(f'xgboost_model_{month}_months.json')
        log_message(f"Модель для {month} месяцев сохранена в 'xgboost_model_{month}_months.json'.")
