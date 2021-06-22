import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import pickle
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image 
from pathlib import Path
from torch.autograd import Variable

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        dataset = load_Dataset()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def load_image(image_file):
	img = Image.open(image_file)
	return img

def load_transformations():
  transformations = pickle.load(open('transformations.pkl', 'rb'))
  return transformations

def load_Dataset():
  dataset = pickle.load(open('dataset.pkl', 'rb'))
  return dataset

def load_Model():
  model = pickle.load(open('model_ResNet.pkl', 'rb'))
  return model

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):

    device = get_default_device()

    dataset = load_Dataset()

    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    prob, preds  = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]


def main():
  st.title("Garbage detector")
  st.subheader("Upload an image to detect the type of residue")
  

  
  image_file = st.file_uploader("Upload your Image",type=['png','jpeg','jpg'])

  if image_file is not None:
    if st.button("Predicting the type of residue"):
      st.image (image_file, caption = 'Uploaded Image.', use_column_width = True)
      image = Image.open(image_file)
      transformations = load_transformations()
      example_image = transformations(image)
      model = load_Model()
      st.write("The image resembles", predict_image(example_image, model))

      
  else:
    st.subheader("You must enter an image to predict the type of waste")
    
if __name__ == '__main__':
	main()
