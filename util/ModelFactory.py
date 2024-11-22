import os
import logging
from models.ModelsMuscle import *
from util.Model import Model

logging.basicConfig(level=logging.INFO)


class ModelFactory:
    def __init__(self, dataset_train, dataset_test):
        self.available_models = [ModelMuscleJulius1, ModelMuscleJulius2, ModelMusclePero, ModelMuscleMarla]
        self.available_model_names = [model.to_string() for model in self.available_models]
        self.models_dict = {model.to_string(): model for model in self.available_models}

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.input_shape, self.output_shape = self.dataset_train.get_io()

    def create_model(self, model_type, criterion=nn.CrossEntropyLoss(), train=True, epochs=20, batch_size=32):
        if model_type not in self.available_model_names:
            logging.error(f"Model type {model_type} is not supported.")
            return

        model = Model(self.get_model_by_name(model_type), self.dataset_train, self.dataset_test, self.input_shape, self.output_shape)

        if train:
            model.train(criterion, epochs, batch_size)
        else:
            logging.warning("Creating an untrained model, please manually train your model later")

        return model

    def create_model_from_save(self, load_path):
        if os.path.exists(load_path):

            parts = load_path.split('_')
            model_name = '_'.join(parts[1:3])

            model = Model(ModelFactory.get_model_by_name(model_name), self.dataset_train, self.dataset_test, self.input_shape, self.output_shape)
            model.model.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
            return model
        else:
            print(f"No saved model found at {load_path}")

    def get_available_model_types(self):
        return self.available_model_names

    def get_model_by_name(self, name):
        return self.models_dict.get(name, None)
