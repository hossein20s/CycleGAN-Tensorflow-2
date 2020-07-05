import datetime
from unittest import TestCase

import numpy
import pandas
import tensorflow
from matplotlib import pyplot
from pandas_datareader import data
from sklearn import preprocessing

from quadratic_gan import QuadraticGAN


def get_y(x):
    return x * x + 0


def is_part_of_line(x):
    initial = (1, 2)
    final = (8, 10)
    if x < initial[0] or x > final[0]:
        return 0
    m = (final[1] - initial[1]) / (final[0] - initial[0])
    b = initial[1] - m * initial[0]
    return m * x + b


def zip_stock():
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2019, 10, 30)
    ticker = 'BNS'
    df1 = data.DataReader(ticker, 'yahoo', start, end)
    df = pandas.DataFrame()
    df['x'] = df1['Adj Close'].index.astype(numpy.int64)
    df['y'] = df1['Adj Close'].values
    values = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    df = pandas.DataFrame(values_scaled)
    return df


class TestQuadraticGAN(TestCase):

    def setUp(self) -> None:
        super().setUp()
        n = len(zip_stock())
        self.gan = QuadraticGAN(target_function=zip_stock, initial_batch_size=3, number_of_batch=3, epochs=30,
                                number_of_interation=2)

    def test_sample_data(self):
        self.gan.target_function = get_y
        data = self.gan.sample_data(number_of_sample=100, scale=10)
        print(data.shape)
        pyplot.scatter(*zip(*data))
        pyplot.show()

    def test_make_generator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1000, 100])
        y = generator(noise, training=False)
        print(y.shape)

        pyplot.scatter(*zip(*y.numpy()))
        pyplot.show()

    def test_make_discriminator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1000, 100])
        y = generator(noise, training=False)
        discriminator = self.gan.make_discriminator_model()
        decision = discriminator(y)
        print(decision)

    def test_run(self):
        self.gan.run(debug=False)
