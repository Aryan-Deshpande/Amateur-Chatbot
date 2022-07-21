from prprocess import bagofwords, tokenize
from model import NeuralNet
import random
import torch
import json

with open('intent.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)


model_state = data["model_state"]

model = NeuralNet(data['input_size'],data['hidden_size'],data['output_size'])
model.load_state_dict(model_state)

tags = data['tags']
while True:
    sentence = input(" - ")

    sentence = tokenize(sentence)
    bag = bagofwords(sentence,data['all_words'])
    bag = bag.reshape(1,bag.shape[0])
    bag = torch.from_numpy(bag)

    output = model(bag)
    _,predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents['intents']:
        if tag == intent['tag']:
            
            print("bot ", {random.choice(intent['responses'])})


