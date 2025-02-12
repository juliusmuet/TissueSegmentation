from util.Dataset import LargeDataset, load_data_with_labels, get_train_test_indices
from util.ModelFactory import ModelFactory
from util.ImageSegmentation import load_and_segment_images
import logging

logging.basicConfig(level=logging.INFO)

"""
Example usage of the framework
"""

# load data and create train and validation data set
spectra, labels, mapping = load_data_with_labels("data/train_data")
logging.info(f"Mapping: {mapping}")
indices_train, indices_validation = get_train_test_indices(spectra.shape[0])
dataset_train = LargeDataset(spectra, labels, indices_train)
dataset_validation = LargeDataset(spectra, labels, indices_validation)

# create model factory based on given datasets
factory = ModelFactory(dataset_train, dataset_validation)

# get list of implemented models
available_models = factory.get_available_model_types()

# select, train and evaluate models
model_1 = factory.create_model(available_models[0])
model_1.evaluate()

# load trained models
#model_1 = factory.create_model_from_save("model_parameters/model_muscle_1_yyyymmdd_hhmmss.pth", "muscle_1")

# image segmentation
load_and_segment_images("data/images", model_1, mapping)

# calculate statistics of model
spectra_test, labels_test, mapping_test = load_data_with_labels("data/test_data")
logging.info(f"Mapping: {mapping_test}")
model_1.calculate_statistics(spectra_test, labels_test, mapping_test, "data/statistics")
