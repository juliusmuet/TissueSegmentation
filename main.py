import os
import numpy as np
from util.Dataset import *
from util.ModelFactory import ModelFactory
from util.LabelVisualiser import load_and_classify_images
import logging

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data and create train and test data set
#spectra, labels, mapping = load_data_with_labels("../../first_training/train_images")
spectra, labels, mapping = load_data_with_labels("../../second_training/train_images")
logging.info(f"Mapping: {mapping}")
indices_train, indices_test = get_train_test_indices(spectra.shape[0])
dataset_train = LargeDataset(spectra, labels, indices_train)
dataset_test = LargeDataset(spectra, labels, indices_test)

# create model factory based on given datasets
factory = ModelFactory(dataset_train, dataset_test)

# get list of implemented models
available_models = factory.get_available_model_types()

"""
# select and train models
model_1 = factory.create_model(available_models[0])
model_2 = factory.create_model(available_models[1])

# evaluate models
model_1.evaluate()
model_2.evaluate()
"""

# load trained models
#model_1 = factory.create_model_from_save("model_parameters/model_muscle_1_20241202_160015.pth", "muscle_1")
#model_2 = factory.create_model_from_save("model_parameters/model_muscle_2_20241202_174920.pth", "muscle_2")
model_1 = factory.create_model_from_save("model_parameters/model_muscle_1_20241220_130424.pth", "muscle_1")
model_2 = factory.create_model_from_save("model_parameters/model_muscle_2_20241220_132043.pth", "muscle_2")

# classify test images
#load_and_classify_images("../../first_training/test_images", model_1, mapping)
#load_and_classify_images("../../first_training/test_images", model_2, mapping)
load_and_classify_images("../../second_training/test_images", model_1, mapping)
load_and_classify_images("../../second_training/test_images", model_2, mapping)


# predict labels
# TODO: actual data for prediction instead of random values
"""
predicted_labels_model_1, probabilities_1 = model_1.predict(np.random.rand(3, 427))
logging.info(f"Predictions for inputted data with {available_models[0]}: \n"
             f"{probabilities_1} : \n"
             f"   {predicted_labels_model_1} : {decode_label_indices(predicted_labels_model_1, mapping)}")
"""