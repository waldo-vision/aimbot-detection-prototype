import torch

def load_data(path_to_data):
    data = torch.load(path_to_data)
    labels = torch.ones(data.shape[0]).unsqueeze(1)
    return data, labels

# torch.load("./no_hacks_data_tensor/clips.pt")
hacks_data, hacks_labels = load_data("./hacks_data_tensor/full_data/hacks_data_tensor_file_100.pt")
no_hacks_data, no_hacks_labels = load_data("./hacks_data_tensor/full_data/hacks_data_tensor_file_100.pt")

# Seperating Training/Testing data into 90%/10% splits
def create_training_testing_sets(data, split=0.9):
    training_data = data[:int(len(data) * split)]
    testing_data = data[int(len(data) * split):]
    return training_data, testing_data

hacks_data_train, hacks_data_test = create_training_testing_sets(hacks_data)
no_hacks_data_train, no_hacks_data_test = create_training_testing_sets(no_hacks_data)
hacks_labels_train, hacks_labels_test = create_training_testing_sets(hacks_labels)
no_hacks_labels_train, no_hacks_labels_test = create_training_testing_sets(no_hacks_labels)

def load_training_data():
    train_data = torch.cat((hacks_data_train, no_hacks_data_train))
    train_labels = torch.cat((hacks_labels_train, no_hacks_labels_train))
    return train_data, train_labels

def load_testing_data():
    test_data = torch.cat((hacks_data_test, no_hacks_data_test))
    test_labels = torch.cat((hacks_labels_test, no_hacks_labels_test))
    return test_data, test_labels