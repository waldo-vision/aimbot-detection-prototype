import torch
from data_loader import load_testing_data
from architectures.RNN import RNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 1
num_epochs = 100

learning_rate = 0.001

input_size = 512
sequence_length = 50
hidden_size = 512
num_layers = 2

# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

model = torch.load("./models/model.pt")

# Loading the model
test_data, test_labels = load_testing_data()

print(test_data.shape)

model.eval() # Model is set to evaluate

# Checking for whether clips are accurately predicted or not
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i in range(len(test_data)):
        curr_test = test_data[i].unsqueeze(0)
        curr_label = test_labels[i][0].item()
        outputs = model(curr_test)

        if(abs(outputs.item()-curr_label) < 0.5):
            print("success" + str(outputs.item()))
            n_correct += 1

        n_samples += 1

print(n_samples)
print("percentage correct: " + str(n_correct/n_samples * 100.))