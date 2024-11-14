import os
from Model import Model
import torch
import logging
from models.ModelsMuscle import *

logging.basicConfig(level=logging.INFO)


class ModelFactory:

    """
    A factory class for creating and loading machine learning models based on a specified dataset.

    This class enables the creation of models, including options to initialize new models,
    load saved model parameters, and manage multiple model types dynamically.

    Attributes:
        available_models (list): List of available model classes.
        available_model_names (list): List of names of available models, derived from each model's `toString` method.
        models_dict (dict): Dictionary mapping model names to their corresponding model classes.
        dataset (Dataset): Dataset used for training and evaluating the model
        input_shape (ndarray): Shape of the original input data
        output_shape (ndarray): Shape of the original output data
    """

    def __init__(self, dataset):
        self.available_models = [ModelMuscleJulius, ModelMusclePero, ModelMuscleMarla]
        self.available_model_names = [model.toString() for model in self.available_models]
        self.models_dict = {model.toString(): model for model in self.available_models}
        self.dataset = dataset
        self.input_shape, self.output_shape = self.dataset.get_io()

    def create_model(self, model_type, criterion=nn.CrossEntropyLoss(), train=True, epochs=20, batch_size=32):
        """
        Creates and optionally trains a model of the specified type.

        Parameters:
            model_type (str): The name of the model type to create.
            criterion (torch.nn.Module): Loss function for training. Default is CrossEntropyLoss.
            train (bool): Whether to train the model immediately. Default is True.
            epochs (int): Number of epochs for training if `train` is True. Default is 20.
            batch_size (int): Batch size for training if `train` is True. Default is 32.

        Returns:
            Model: A trained or untrained Model instance, or None if `model_type` is invalid.
        """
        if model_type not in self.available_model_names:
            logging.error(f"Model type {model_type} is currently not supported or misspelled.")
            return

        model = Model(self.get_model_by_name(model_type), self.input_shape, self.output_shape)

        if train:
            model.train(self.dataset, criterion, epochs, batch_size)
        else:
            logging.warning("Creating an untrained model, please manually train your model later")

        return model

    def create_model_from_save(self, load_path):
        """
        Loads a saved model's parameters from a file.

        Parameters:
            load_path (str): Path to the saved model file.

        Returns:
           Model: A Model instance with loaded parameters if the file exists; otherwise, None.
        """
        if os.path.exists(load_path):
            # get model name
            parts = load_path.split('_')
            model_name = '_'.join(parts[1:3])
            model = Model(ModelFactory.get_model_by_name(model_name), self.input_shape, self.output_shape)
            model.model.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
            return model
        else:
            print(f"No saved model found at {load_path}")

    def get_available_model_types(self):
        return self.available_model_names

    def get_model_by_name(self, name):
        return self.models_dict.get(name, None)
