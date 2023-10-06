import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('Chicago_hotels.csv', delimiter=';', decimal=",")
    data['date'] = pd.to_datetime(data['date1'] + ' ' + data['year'].astype(str))

    # Заменяем пустые значения на NaN и преобразуем столбец в числовой тип
    data['cost'] = pd.to_numeric(data['cost'].str.replace(',', '.'), errors='coerce')

    # Убираем лишние столбцы и переименовываем столбец с данными о средней цене
    data = data[['date', 'cost']]
    data = data.rename(columns={'cost': 'Average Daily Rate (in $)'})

    # Устанавливаем дату в качестве индекса
    data.set_index('date', inplace=True)

    # Построим график исходного временного ряда
    plt.figure(figsize=(14, 7))  # Увеличиваем размер графика
    data['Average Daily Rate (in $)'].plot(marker='o', linestyle='-')
    plt.title('Исходный временной ряд средней цены')
    plt.xlabel('Дата')
    plt.ylabel('Средняя цена (в $)')
    plt.grid(True)  # Добавляем сетку
    plt.xticks(rotation=45)  # Поворачиваем метки по оси X для лучшей читаемости
    plt.tight_layout()  # Улучшает отображение меток на графике
    plt.show()


if __name__ == '__main__':
    main()
