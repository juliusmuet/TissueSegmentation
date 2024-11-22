import numpy as np
from util.Dataset import LargeDataset, get_train_test_indices
from util.ModelFactory import ModelFactory
import logging

logging.basicConfig(level=logging.INFO)


spectra = np.random.rand(1000, 420) # Beispiel Spektrum
labels = np.random.randint(0, 2, size=(1000, 2))    # Beispiel Labels

indices_train, indices_test = get_train_test_indices(spectra.shape[0])
dataset_train = LargeDataset(spectra, labels, indices_train)
dataset_test = LargeDataset(spectra, labels, indices_test)

factory = ModelFactory(dataset_train, dataset_test)  # create model factory based on given datasets

available_models = factory.get_available_model_types()  # get list of implemented models

# select and train model
model_muscle_julius_1 = factory.create_model(available_models[0])
model_muscle_julius_2 = factory.create_model(available_models[1])
#model_muscle_pero = factory.create_model(available_models[2])
#model_muscle_marla = factory.create_model(available_models[3])

# evaluate model
model_muscle_julius_1.evaluate()
model_muscle_julius_2.evaluate()
#model_muscle_pero.evaluate()
#model_muscle_marla.evaluate()

# predict labels
logging.info(f"Predicted labels for inputted data with {available_models[0]}: {model_muscle_julius_1.predict(np.random.rand(3, 420))}")
logging.info(f"Predicted labels for inputted data with {available_models[1]}: {model_muscle_julius_2.predict(np.random.rand(3, 420))}")
