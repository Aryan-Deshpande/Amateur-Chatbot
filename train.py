import json
from prprocess import tokenize, stemming, bagofwords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intent.json', 'r') as f:
    filecontent = json.load(f)
#print(intents)

tags = []
all_words = []
pat = []

for content in filecontent['intents']:
    tag = content['tag']
    tags.append(tag)
    
    for sentence in content['patterns']:
        tk = tokenize(sentence)
        all_words.extend(tk)
    
        pat.append((tag,tk))

avoid = ['!','.','!!','?'] # just to avoid words #

all_words = [stemming(w) for w in all_words if w not in avoid]

xtrain = []
ytrain = []

for (tag, pattern) in pat:
    bag = bagofwords(tk, all_words)
    xtrain.append(bag)

    label = tags.index(tag) # creating labels
    # cross entropy loss
    ytrain.append(label) # use use 1 hot encoded vector #

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(xtrain)
        self.x_data = xtrain
        self.y_data = ytrain
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)

# hyperparameters 
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(xtrain[0])
learning_rate = 0.001
nepochs = 2000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# create loss & optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(nepochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backprop and optimizer
            #empty gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 == 0:
        print(f"epoch {epoch+1}/{nepochs}, loss={loss.item():4f}")

print(epoch, loss.item())

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags
}

FILE  = "data.pth"
torch.save(data,FILE)