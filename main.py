import numpy as np
from Dataset import Dataset
from ModelFactory import ModelFactory


# TODO: load actual data
spectra = np.random.rand(1000, 420) # Beispiel Spektrum
labels = np.random.randint(0, 2, size=(1000, 2))    # Beispiel Labels
dataset = Dataset(spectra, labels)

factory = ModelFactory(dataset)  # create models based on given dataset

available_models = factory.get_available_model_types()  # get a list of implemented model types to chose from

# select and train model
model_muscle_julius = factory.create_model(available_models[0])
#model_muscle_pero = factory.create_model(available_models[1])
#model_muscle_marla = factory.create_model(available_models[2])

# evaluate the model
model_muscle_julius.evaluate()
#model_muscle_pero.evaluate()
#model_muscle_marla.evaluate()

# predict labels
print(model_muscle_julius.predict(np.random.rand(3, 420)))  # TODO: load actual data
