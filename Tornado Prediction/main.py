import pandas as p
import numpy as n
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def random_data_CSV():
    n.random.seed(42)
    num_samples = 100

    temperature = n.random.randint(20, 35, num_samples)
    humidity = n.random.randint(40, 80, num_samples)
    atmospheric_pressure = n.random.randint(980, 1050, num_samples)
    wind_speed = n.random.randint(0, 40, num_samples)
    cloud_cover = n.random.randint(0, 100, num_samples)

    tornado = n.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Simulate tornado (1) and not a tornado (0)

    data = p.DataFrame({'temperature': temperature, 'humidity': humidity, 'atmospheric_pressure': atmospheric_pressure,
                        'wind_speed': wind_speed, 'cloud_cover': cloud_cover, 'tornado': tornado})

    data.to_csv('Tornado_Random_Data.csv', index=False)


def Prediction():
    data = p.read_csv('Tornado_Random_Data.csv')
    X = data[['temperature', 'humidity', 'atmospheric_pressure', 'wind_speed', 'cloud_cover']]
    y = data['tornado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    temp = int(input("Enter the value of temperature in Celsius = "))
    humid = int(input("Enter the value of humidity = "))
    atmospheric_p = int(input("Enter the value of atmospheric pressure = "))
    wind_s = int(input("Enter the value of wind speed in MPH = "))
    cloud_c = int(input("Enter the value of cloud cover = "))

    new_data = p.DataFrame({'temperature': [temp], 'humidity': [humid], 'atmospheric_pressure': [atmospheric_p],
                            'wind_speed': [wind_s], 'cloud_cover': [cloud_c]})

    predicted_tornado = model.predict(new_data)

    print(
        f'Will there be a tornado for Temperature = {new_data["temperature"].iloc[0]}, '
        f'Humidity = {new_data["humidity"].iloc[0]}, '
        f'Atmospheric Pressure = {new_data["atmospheric_pressure"].iloc[0]}, '
        f'Wind Speed = {new_data["wind_speed"].iloc[0]} and '
        f'Cloud Cover = {new_data["cloud_cover"].iloc[0]}'
        f' ?'f'{" Yes" if predicted_tornado[0] else " No"}')


def repeat():
    while True:
        print("Welcome to the Simple Tornado Prediction Software")
        print("        Developed by Talha Khalid\n")
        print("** Important Note ** \nFor making this software, Talha Khalid has used "
              "random generated data for temperature, humidity, atmospheric pressure, wind speed and cloud cover"
              " to train the model.")

        Prediction()

        user = input("Do you want to use it again. Press Y/N = ").lower()
        print("\n")
        if user != 'y' and user == "n":
            print("Thanks for using. Have a lovely day :)")
            break

        elif user != 'n' and user != 'y':
            print("You neither press Y nor N so I am Shutting Down the software..."
                  " Thanks for using. Have a lovely day :)")
            break


repeat()
