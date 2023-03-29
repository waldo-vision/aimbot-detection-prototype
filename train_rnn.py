import torch
import torch.nn as nn
from architectures.RNN import RNN
from data_loader import load_training_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 1
num_epochs = 100

learning_rate = 0.001

input_size = 512
sequence_length = 50
hidden_size = 512
num_layers = 2

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # Creating instance of LSTM model

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

train_data, train_labels = load_training_data()

model.train() # set model to training mode 
for epoch in range(num_epochs):
    images = train_data         # using our set of images
    labels = train_labels       # using our set of labels
    labels = labels.to(device)  # uploading onto CPU/GPU

    random_shuffle = torch.randperm(images.size()[0])  # shuffling the data for every epoch
    images = images[random_shuffle]
    labels = labels[random_shuffle]
    
    # Forward pass
    outputs = model(images)     # perform a forward pass 
    loss = criterion(outputs, labels) # calculate loss/error
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') # print results for every epoch

    torch.save(model, "./models/model.pt") # incrementally save model
