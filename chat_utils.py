#import argparse
#import os
#import sys
#import time
#import re
import json
import random

#import numpy as np
import torch
#from torch.optim import Adam
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision import transforms
import torch.onnx
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#import utils
#from transformer_net import TransformerNet
#from vgg import Vgg16
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#@st.cache
def load_model(model_path):
    print('load model')

    #FILE = "data.pth"
    data = torch.load(model_path)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, all_words, tags


#@st.cache
def chatting(model, sentence, all_words, tags):

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    with torch.no_grad():
        output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                output = random.choice(intent['responses'])
    else:
        output = "I do not understand..."

    return output
