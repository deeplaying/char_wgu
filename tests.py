import tensorflow as tf
from model import SimpleLSTMNetwork
from prepare_data import CharData

def test_data_wrapper():
    data = CharData('data')
    print(data.length_of_text)


def test_building_graph():

    model = SimpleLSTMNetwork(X)
    

def main():

    print('testing data wrapper. ')
    test_data_wrapper()
    pass


if __name__ == '__main__':
    main()