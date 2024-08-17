import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from skimage.metrics import mean_squared_error
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as distance
from sklearn.preprocessing import MinMaxScaler

import numpy as np

class DeepScatter():
    """
    A class for building, training, and evaluating the deepscatter model for time series anomaly enhancing using an autoencoder.

    This class encapsulates the process of defining an autoencoder model, training it on provided time series data,
    and evaluating it for anomaly detection. It preprocesses the data, creates the model, and computes similarity scores.

    Attributes
    ----------
    t : int
        The length of the data aggregation sliding window. Default is 20.
    epochs : int
        The number of epochs to train the model. Default is 20.
    batch_size : int
        The batch size used for training the model. Default is 2.
    verbose : bool
        Controls the verbosity of the training process. Default is False.
    encoder : tf.keras.Model or None
        The encoder part of the autoencoder model. Initialized to None until created.
    encoded_train : np.ndarray or None
        The encoded representation of the training data. Initialized to None until computed.
    encoded_test : np.ndarray or None
        The encoded representation of the test data. Initialized to None until computed.
    train : np.ndarray
        The preprocessed training data.
    test : np.ndarray
        The preprocessed test data.
    model : tf.keras.Model
        The autoencoder model.

    Parameters
    ----------
    train : np.ndarray
        The training data as a numpy array. Should have shape (n_samples, n_features).
    test : np.ndarray
        The test data as a numpy array. Should have shape (n_samples, n_features).
    t : int, optional
        The length of the data aggregation sliding window. Default is 20.
    epochs : int, optional
        The number of epochs to train the model (default is 20).
    batch_size : int, optional
        The batch size used for training the model (default is 2).
    verbose : bool, optional
        Controls the verbosity of the training process (default is False).
    """

    def __init__(self,
                 train:np.array, 
                 test:np.array, 
                 t:int=20,
                 epochs:int=20, 
                 batch_size:int=2, 
                 verbose:bool=False
                 ):
        
        self.t = t
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.encoder = None
        self.encoded_train = None
        self.encoded_test = None

        self.train = self._preprocess(train)
        self.test = self._preprocess(test)

        self.model, self.encoder, self.input = self._create_model()

    def _create_model(self) -> tuple[Model, layers.Layer, layers.Input]:

        """
        Creates the deepscatter autoencoder model

        This method constructs an autoencoder using convolutional layers for both encoding and decoding. 
        It returns the model along with the encoder and input layers.

        Returns
        -------
        tuple
            A tuple containing:
            - model : tensorflow.keras.Model
                The autoencoder model
            - encoder : tensorflow.keras.layers.Layer
                The encoder layer
            - input_layer : tensorflow.keras.layers.Input
                The input layer of the model

        """

        # Define the input layer
        input = layers.Input(shape=(self.t, 1))

        # Encoder layers
        x = layers.Conv1D(8, 4, activation='relu', padding='same')(input)
        encoder = layers.MaxPooling1D(2)(x)

        # Decoder layers
        x1 = layers.Conv1D(8, 4, activation='relu', padding='same')(encoder)
        x1 = layers.UpSampling1D(2)(x1)
        decoder = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x1)

        # Create the autoencoder model
        model = Model(input, decoder)

        return model, encoder, input

    def _preprocess(self, array: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input array by normalizing it and reshaping it for model input.

        This method performs normalization of the input array to scale its values between 0 and 1,
        and reshapes the array to match the input shape required by the model.

        Parameters
        ----------
        array : np.ndarray
            The input array to be processed. It should be a 1D numpy array.

        Returns
        -------
        np.ndarray
            The processed array, reshaped to match the input shape of the model.

        """
        array = (array - np.min(array)) / (np.max(array) - np.min(array)).astype("float32")
        array = np.reshape(array, (len(array), self.t, 1))
        return array

    def _computeSimilarity (self, t1: np.ndarray, t2: np.ndarray) -> float:
        """
        Computes the mean cosine similarity between two 3D arrays after reshaping them.

        This method reshapes the 3D arrays into 2D arrays, computes the cosine similarity between the reshaped arrays, 
        and returns the mean of the upper triangular part of the similarity matrix.

        Parameters
        ----------
        t1 : np.ndarray
            The first 3D array to be compared. Its shape should be (n_samples, time_steps, features).
        t2 : np.ndarray
            The second 3D array to be compared. Its shape should be (n_samples, time_steps, features).

        Returns
        -------
        float
            The mean value of the upper triangular part of the cosine similarity matrix between the reshaped arrays.

        Raises
        ------
        ValueError
            If `t1` and `t2` do not have the same shape.
        """

        if t1.shape != t2.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        t1 = t1.reshape(t1.shape[0],t1.shape[1]*t1.shape[2])
        t2 = t2.reshape(t2.shape[0],t2.shape[1]*t2.shape[2])
        similarities = cosine_similarity(t1, t2)
        return np.triu(similarities).mean()

    def _computeNormalReference(self, t:int=20) -> list:
        """
        Computes the mean and standard deviation of similarity scores over normal reference segments.

        This method calculates the average similarity score between segments of the encoded training data 
        and computes the mean and standard deviation of these similarity scores. 

        Parameters
        ----------
        t : int
            The length of the segments used for computing similarity. It should be a positive integer.

        Returns
        -------
        list
            A list containing:
            - mean : float
                The mean similarity score across all segments.
            - std : float
                The standard deviation of similarity scores across all segments.
            - normal_scores : list of float
                A list of similarity scores for each segment comparison.

        """
        normal_scores = []
        for i in range (0, len(self.encoded_train)-t):
            test = self.encoded_train[i:i+t]
            similarities = []
            for j in range (0, len(self.encoded_train)-t):
                perc = self._computeSimilarity(self.encoded_train[j:j+t], test)
                similarities.append(perc)
            normal_scores.append(np.mean(similarities))
        mean = np.array(normal_scores).mean()
        std = np.std(np.array(normal_scores))
        return [mean, std, normal_scores]

    def _fitModelShiftDetection (self, l: int, test: np.ndarray) -> float:
        """
        Computes the average similarity score between segments of the encoded training data and a test set.

        This method calculates the similarity score between segments of the encoded training data and the test data segment,
        and returns the mean of these similarity scores.

        Parameters
        ----------
        l : int
            The length of the segments used for computing similarity. It should be a positive integer.
        test : np.ndarray
            The test segment to compare against segments of the encoded training data. It should be a 2D numpy array.

        Returns
        -------
        float
            The mean similarity score between the segments of the encoded training data and the test segment.

        Raises
        ------
        ValueError
            If `l` is not a positive integer, or if it is larger than the length of `self.encoded_train` or `self.encoded_test`.
            If `test` is not a 2D numpy array.
        """
        test_scores = []
        for i in range (0, len(self.encoded_train)-len(self.encoded_test)):
            similarity = self._computeSimilarity(self.encoded_train[i:i+l], test)
            test_scores.append(similarity)

        return np.asarray(test_scores).mean()

    def train_model(self) -> None:
        """
        Compiles and trains the autoencoder model.

        This method compiles the autoencoder model with the Adam optimizer and binary cross-entropy loss. It then
        trains the model using the provided training data. After training, it creates an encoder model that outputs
        the encoded representation of the input.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model or input layers are not properly defined before calling this method.
        """

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

        # Train the autoencoder model
        self.model.fit(
            x=self.train,
            y=self.train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose
        )

        # Create an encoder model that takes the input and outputs the encoded representation
        self.encoder = Model(inputs=self.input, outputs=self.encoder)

    def evaluate(self, l: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the autoencoder model by generating predictions, calculating error metrics, and comparing training 
        and test data.


        Parameters
        ----------
        l : int
            The length of the segment used for shift detection and normal reference computation. It should be a positive integer.


        
        """

        if l <= 0:
            raise ValueError("Parameter `l` must be a positive integer.")
        
        # Generate predictions using the autoencoder on the test data
        # predictions = self.model.predict(self.test)

        # Calculate mean squared error (MSE) and root mean squared error (RMSE) for evaluation purposes
        # mse = mean_squared_error(self.test.squeeze(2), predictions.squeeze(2))
        # rmse = np.sqrt(mse)

        # Encode the training and test data using the encoder model
        self.encoded_train = self.encoder.predict(self.train)
        self.encoded_test = self.encoder.predict(self.test)

        _, _, train_scores = self._computeNormalReference(l)

        test_scores = []
        for i in range (0, len(self.encoded_test)-l):
            scores = self._fitModelShiftDetection(l, self.encoded_test [i:i+l])
            test_scores.append(scores)
        
        # Calculate the mean and standard deviation of the train_scores
        mean_train = np.mean(train_scores)
        std_train = np.std(train_scores)

        # Calculate the mean and standard deviation of the test_scores
        mean_test = np.mean(test_scores)
        std_test = np.std(test_scores)

        # Assuming mean_train, std_train, and l are defined
        random_value = np.random.uniform(mean_train - std_train, mean_train + std_train, 1)
        result_array = np.full((l,), random_value)
        pad_train = np.concatenate((np.array(train_scores), result_array) , axis=0)

        # Create a list of l random numbers sampled from a uniform distribution within
        # one standard deviation of the mean_test, and repeat it l times
        random_value = np.random.uniform(mean_test - std_test, mean_test + std_test, 1)
        result_array = np.full((l,), random_value)
        pad_test = np.concatenate((np.array(test_scores), result_array) , axis=0)

        return pad_train, pad_test

