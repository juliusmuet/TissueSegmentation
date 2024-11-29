import os
import numpy as np
from util.Dataset import *
from util.ModelFactory import ModelFactory
import logging

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data and create train and test data set
# TODO: change to actual directory names
spectra, labels = load_and_merge_two_directories("../TestData/muscle", "../TestData/other")
indices_train, indices_test = get_train_test_indices(spectra.shape[0])
dataset_train = LargeDataset(spectra, labels, indices_train)
dataset_test = LargeDataset(spectra, labels, indices_test)

# create model factory based on given datasets
factory = ModelFactory(dataset_train, dataset_test)

# get list of implemented models
available_models = factory.get_available_model_types()

# select and train models
model_muscle_julius_1 = factory.create_model(available_models[0])
model_muscle_julius_2 = factory.create_model(available_models[1])
#model_muscle_pero = factory.create_model(available_models[2])
#model_muscle_marla = factory.create_model(available_models[3])

# evaluate models
model_muscle_julius_1.evaluate()
model_muscle_julius_2.evaluate()
#model_muscle_pero.evaluate()
#model_muscle_marla.evaluate()

# predict labels
# TODO: actual data for prediction instead of random values
#logging.info(f"Predicted labels for inputted data with {available_models[0]}: {model_muscle_julius_1.predict(np.random.rand(3, 427))}")
#logging.info(f"Predicted labels for inputted data with {available_models[1]}: {model_muscle_julius_2.predict(np.random.rand(3, 427))}")
