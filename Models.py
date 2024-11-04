import tensorflow as tf;
from tensorflow.keras.applications import ResNet50, ResNet50V2;
from tensorflow.keras import layers, models;
import constants;
import matplotlib.pyplot as plt;
import numpy as np;
from setup import encodeLabel, decodeLabel;

class Model:
    resnet50: ResNet50 = None;
    resnet50V2: ResNet50V2 = None;
    resnet50V2_regressor: ResNet50V2 = None;

    def __init__(self, resnet50: ResNet50 = None, resnet50V2: ResNet50V2 = None):
        '''
        Constructor: Load the pretrained models
        '''
        if (resnet50 == None):
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
        else:
            self.resnet50 = resnet50;
        
        # Freeze all layers in the base model
        self.resnet50.trainable = False;

        if (resnet50V2 == None):
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
        else:
            self.resnet50V2 = resnet50V2;

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
        self.resnet50.save("resnet50-pretrained.h5");
        self.resnet50V2.save("resnet50V2-pretrained.h5");

    def train(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, modelName: str, isRegression: bool = False):
        '''
        Train a specific classifier.
        '''
        if (isRegression != True):
            # Encode the labels only if it is a classification problem
            y_train_encoded = encodeLabel(y_train);
            y_val_encoded = encodeLabel(y_val);
            print(f"Encoded Labels in training set: {np.unique(y_train_encoded)}");
            print(f"Encoded Labels in validation set: {np.unique(y_val_encoded)}");
        history = model.fit(
            X_train, 
            y_train_encoded if isRegression != True else y_train, 
            epochs=constants.epoch, 
            validation_data=(
                X_val, 
                y_val_encoded if isRegression != True else y_val
            )
        );
        # Save the models
        if (modelName == constants.RESNET50):
            model.save("resnet50-posttrained.h5");
            self.resnet50 = model;
        elif (modelName == constants.RESNET50V2):
            if (isRegression == True):
                model.save("resnet50V2-regression-posttrained.h5");
                self.resnet50V2_regressor = model;
            else:
                model.save("resnet50V2-posttrained.h5");
                self.resnet50V2 = model;

        self.plotLearningCurve(history);
        return history;

    def plotLearningCurve(self, history):
        '''
        Plot training & validation accuracy values
        '''
        # deduce the string of the metric to be used
        k = '';
        # accuracy metric
        if 'accuracy' in history.history :
            plt.title('Learning Curve - Model Accuracy');
            plt.ylabel('Accuracy');
            k = 'accuracy';    

        elif 'acc' in history.history :
            plt.title('Learning Curve - Model Accuracy');
            plt.ylabel('Accuracy');
            k = 'acc';
        
        # mean squared error metric
        elif 'mae' in history.history:
            plt.title('Learning Curve - Model Mean Squared Error');
            plt.ylabel('MAE');
            k = 'mae';
        
        # plot
        plt.plot(history.history[k]);
        plt.plot(history.history['val_'+ k])
        plt.xlabel('Epoch');
        plt.legend(['Train', 'Validation'], loc='upper left');
        plt.show();
    
        # loss plot
        plt.plot(history.history['loss']);
        plt.plot(history.history['val_loss']);
        plt.title('Learning Curve - Model loss');
        plt.ylabel('Loss');
        plt.xlabel('Epoch');
        plt.legend(['Train', 'Validation'], loc='upper left');
        plt.show();

    def evaluate(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        '''
        Calculate binary cross-entropy error on each dataset
        '''
        train_loss, train_customMetric = model.evaluate(X_train, y_train, verbose=0);
        val_loss, val_customMetric = model.evaluate(X_val, y_val, verbose=0);
        test_loss, test_customMetric = model.evaluate(X_test, y_test, verbose=0);
        print(f"Training Binary Cross-Entropy error: {train_loss}");
        print(f"Validation Binary Cross-Entropy error: {val_loss}");
        print(f"Testing Binary Cross-Entropy error: {test_loss}");
        return ([train_loss, train_customMetric], [val_loss, val_customMetric], [test_loss, test_customMetric]);


    def predict(self, model, X_test: np.ndarray, isRegression: bool = False):
        y_pred = model.predict(X_test);
        print(f"Number of Predictions made: {len(y_pred)}");
        if (isRegression != True):
            # Classification
            print(f"Unique Labels in Prediction: {np.unique(decodeLabel(y_pred))}");
        else:
            # Regression
            print(f"Number of Unique Labels in Prediction: {len(np.unique(y_pred))}");
        return y_pred;

    def transferLearning(self, baseModel=None):
        '''
        Transfer our model's learning from a classification approach to a regression approach.
        '''
        if (baseModel != None):
            # Build the regression head        
            model = models.Sequential();
            model.add(baseModel);
            # Add a dense layer with ReLU activation (optional)
            model.add(layers.Dense(1024, activation='relu'));
            # Output layer with 4 units for bounding box coordinates
            # (no activation or use linear activation, suitable for regression)
            model.add(layers.Dense(4, activation='linear'));
            # Compile the model with Mean Squared Error (MSE) loss
            model.compile(optimizer='adam', loss='mse', metrics=['mae']);
            # Save the model to a .h5 file
            model.save("resnet50V2-regression-posttrained.h5");

            self.resnet50V2_regressor = model;
            return model;
        return baseModel;
