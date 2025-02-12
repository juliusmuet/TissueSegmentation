import os
import logging
from models.ModelsMuscle import *
from util.Dataset import LargeDataset
from util.Model import Model

logging.basicConfig(level=logging.INFO)


class ModelFactory:
    """
    A factory class for creating and managing machine learning models with MULTIPLE OUTPUT NEURONS.
    It supports model creation, training, loading from saved checkpoints, and retrieving available models.

    Attributes:
        available_models (list): List of available model classes.
        models_dict (dict): Dictionary mapping model names to their respective classes.
        dataset_train (Dataset): Training dataset object.
        dataset_test (Dataset): Testing dataset object.
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output data (labels).
    """

    def __init__(self, dataset_train: LargeDataset, dataset_test: LargeDataset):
        """
        Initializes the ModelFactory with training and testing datasets.

        Args:
            dataset_train (LargeDataset): The training dataset object.
            dataset_test (LargeDataset): The testing dataset object.
        """
        self.available_models = [ModelMuscle1, ModelMuscle2, ModelMuscle3, ModelMuscle4]
        self.models_dict = {model.to_string(): model for model in self.available_models}

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.input_shape, self.output_shape = self.dataset_train.get_io()

    def create_model(self, model_type: str, criterion=nn.CrossEntropyLoss(), train: bool = True, epochs: int = 10, batch_size: int = 32):
        """
        Creates and optionally trains a model based on the specified type.

        Args:
            model_type (str): The name of the model to create.
            criterion (nn.Module, optional): The loss function to use for training. Defaults to CrossEntropyLoss.
            train (bool, optional): If True, trains the model immediately after creation. Defaults to True.
            epochs (int, optional): Number of epochs for training. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 32.

        Returns:
            Model: The created model instance. If training is enabled, the model is trained.
                   Returns None if the model type is not available.
        """
        if model_type not in self.models_dict.keys():
            logging.error(f"Model type {model_type} is not supported.")
            return

        model = Model(self.get_model_by_name(model_type), self.dataset_train, self.dataset_test, self.input_shape, self.output_shape)

        if train:
            model.train(criterion, epochs, batch_size)
        else:
            logging.warning("Creating an untrained model, please manually train your model later")

        return model

    def create_model_from_save(self, load_path: str, model_name: str):
        """
        Creates a model specified by model_name by loading its parameters from a saved file.

        Args:
            load_path (str): Path to the file containing the saved model parameters.
            model_name (str): Name of the model to load

        Returns:
            Model: The loaded model instance with parameters restored.
                   Returns None if the file does not exist.
        """
        if os.path.exists(load_path):
            model = Model(self.get_model_by_name(model_name), self.dataset_train, self.dataset_test, self.input_shape, self.output_shape)
            model.model.load_state_dict(torch.load(load_path))
            logging.info(f"Model parameters loaded from {load_path}")
            return model
        else:
            logging.warning(f"No saved model found at {load_path}")

    def get_available_model_types(self):
        """
        Retrieves the list of available model names.

        Returns:
            list: A list of strings representing the names of the available models.
        """
        return list(self.models_dict.keys())

    def get_model_by_name(self, name: str):
        """
        Retrieves a model class by its name.

        Args:
            name (str): The name of the model.

        Returns:
            type: The model class corresponding to the given name, or None if not found.
        """
        return self.models_dict.get(name, None)
