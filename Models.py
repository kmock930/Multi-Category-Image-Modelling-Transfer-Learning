import tensorflow as tf;
from tensorflow.keras.applications import ResNet50, ResNet50V2;
from tensorflow.keras import layers, models;
import constants;
import matplotlib.pyplot as plt;
import numpy as np;
from setup import encodeLabel, decodeLabel;

class Model:
    resnet50: ResNet50;
    resnet50V2: ResNet50V2;

    def __init__(self):
        '''
        Constructor: Load the pretrained models
        '''
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
        self.resnet50 = tf.keras.applications.ResNet50(
            include_top=False, # Remove fully-connected layers
            weights='imagenet',
            input_tensor=None,
            input_shape=constants.input_shape,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        );

        # Freeze all layers in the base model
        self.resnet50.trainable = False

        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2
        self.resnet50V2 = tf.keras.applications.ResNet50V2(
            include_top=False, # Remove fully-connected layers
            weights='imagenet',
            input_tensor=None,
            input_shape=constants.input_shape,
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        );

        # Freeze all layers in the base model
        self.resnet50V2.trainable = False
    
    def build_classifier(self):
        '''
        Build the classifiers
        '''
        # ResNet50
        inputs = tf.keras.Input(shape=constants.input_shape);
        x = self.resnet50(inputs, training=False);
        x = layers.GlobalAveragePooling2D()(x);
        x = layers.Dense(1024, activation='relu')(x);
        outputs = layers.Dense(1, activation='sigmoid')(x);
        self.resnet50 = models.Model(inputs, outputs);

        # ResNet50V2
        x = self.resnet50V2(inputs, training=False);
        x = layers.GlobalAveragePooling2D()(x);
        x = layers.Dense(1024, activation='relu')(x);
        outputs = layers.Dense(1, activation='sigmoid')(x);
        self.resnet50V2 = models.Model(inputs, outputs);

        # Compile the Models
        self.resnet50.compile(optimizer=constants.optimizer, loss=constants.loss, metrics=constants.metrics);
        self.resnet50V2.compile(optimizer=constants.optimizer, loss=constants.loss, metrics=constants.metrics);

        # Save the models
        np.save("resnet50-pretrained.npy", self.resnet50);
        np.save("resnet50V2-pretrained.npy", self.resnet50V2);

    def train(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        '''
        Train a specific classifier.
        '''
        y_train_encoded = encodeLabel(y_train);
        y_val_encoded = encodeLabel(y_val);
        print(f"Encoded Labels in training set: {np.unique(y_train_encoded)}");
        print(f"Encoded Labels in validation set: {np.unique(y_val_encoded)}");
        history = model.fit(
            X_train, 
            y_train_encoded, 
            epochs=constants.epoch, 
            validation_data=(
                X_val, 
                y_val_encoded
            )
        );
        # Save the models
        np.save("resnet50-posttrained.npy", self.resnet50);
        np.save("resnet50V2-posttrained.npy", self.resnet50V2);

        self.plotLearningCurve(history);
        return history;

    def plotLearningCurve(self, history):
        '''
        Plot training & validation accuracy values
        '''
        plt.plot(history.history['accuracy']);
        plt.title('Learning Curve - Model vs Accuracy');
        plt.ylabel('Accuracy');
        plt.xlabel('Epoch');
        plt.legend(['Train'], loc='upper left');
        plt.show();

    def predict(self, model, X_test: np.ndarray):
        y_pred = model.predict(X_test);
        print(f"Number of Predictions made: {len(y_pred)}");
        print(f"Unique Labels in Prediction: {np.unique(decodeLabel(y_pred))}");
        return y_pred;