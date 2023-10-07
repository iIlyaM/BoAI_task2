import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


def main():
    data = pd.read_csv('Chicago_hotels.csv', delimiter=';', decimal=",")
    data['date'] = pd.to_datetime(data['date1'] + ' ' + data['year'].astype(str))

    data['cost'] = pd.to_numeric(data['cost'].str.replace(',', '.'), errors='coerce')

    data = data.dropna(subset=['cost'])

    data = data[['date', 'cost']]
    data = data.rename(columns={'cost': 'Average Daily Rate (in $)'})

    data.set_index('date', inplace=True)

    plt.figure(figsize=(12, 6))
    data['Average Daily Rate (in $)'].plot(marker='o', linestyle='-')
    plt.title('Исходный временной ряд')
    plt.xlabel('Дата')
    plt.ylabel('Средняя цена (в $)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    decompose = seasonal_decompose(data)
    decompose.plot()

    plt.show()

    train = data[:'1999-12']
    test = data['2000-01':]

    plt.plot(train, label='Обучающая', color='black')
    plt.plot(test, label='Тестирующая', color='red')
    plt.legend(title='', loc='upper left', fontsize=14)

    plt.title('Разделение данных о средней цене на обучающую и тестовую выборки')
    plt.ylabel('Средняя цена')
    plt.xlabel('Месяцы')
    plt.grid()
    plt.show()

    model = SARIMAX(train,
                    order=(3, 0, 0),
                    seasonal_order=(0, 1, 0, 12))

    result = model.fit()

    start = len(train)

    end = len(train) + len(test) - 1

    predictions = result.predict(start, end)

    plt.plot(train, label="Обучающая выборка", color="black")
    plt.plot(test, label="Тестовая выборка", color="red")
    plt.plot(predictions, label="Тестовый прогноз", color="green")
    plt.legend(title='', loc='upper left', fontsize=12)

    plt.title("Обучающая выборка, тестовая выборка и тестовый прогноз")
    plt.ylabel('Средняя цена')
    plt.xlabel('Месяцы')
    plt.grid()
    plt.show()

    print(mean_squared_error(test, predictions))

    print(np.sqrt(mean_squared_error(test, predictions)))

    start = len(data)
    end = (len(data) - 1) + 8
    forecast = result.predict(start, end)

    plt.plot(data, label="Фактические данные", color='black')
    plt.plot(forecast, label="Прогноз", color='blue')
    plt.legend(title='', loc='upper left', fontsize=14)

    plt.title('Фактические данные и прогноз на будущее')
    plt.ylabel('Цены')
    plt.xlabel('Месяцы')
    plt.grid()
    plt.show()

    print(forecast)


if __name__ == '__main__':
    main()
