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
    st.subheader("You must upload an image to predict the type of residue")
    
if __name__ == '__main__':
	main()
