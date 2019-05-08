from generate_data import create_trace

def create_data(train_samples_number = 100, test_samples_number = 10):
    create_trace('train', train_samples_number)
    create_trace('test', test_samples_number)

if __name__ == "__main__":

    print('Creating data ...... ')
    create_data()
