import os

from load_dataset import *
from train import *
from test import *

CURR_FILE = os.path.abspath(__file__)

SRC_DIR = os.path.dirname(CURR_FILE)
ROOT_DIR = os.path.dirname(SRC_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

if not os.path.isdir(MODEL_DIR):
	os.makedirs(MODEL_DIR)

if not os.path.isdir(DATA_DIR):
	os.makedirs(DATA_DIR)	

#Load Dataset
ds_train, ds_test = load_dataset()

#Train model
model = train(ds_train)

#Test model
results = test(model, ds_test)

#Save model
model.save_weights(os.path.join(MODEL_DIR,'weights'))

print("Loss: ", results[0])
print("Accuracy: ", results[1])

with open(os.path.join(ROOT_DIR, 'results.txt'), 'w') as fin:
	fin.write("Loss: %s\n"%(str(results[0])))
	fin.write("Accuracy: %s\n"%(str(results[1])))
