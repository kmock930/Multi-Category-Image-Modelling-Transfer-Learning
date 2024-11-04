import tensorflow as tf;
from tensorflow.keras.applications import ResNet50, ResNet50V2;
from tensorflow.keras.utils import to_categorical;
from tensorflow.keras import layers, models, Input;
import constants;
import matplotlib.pyplot as plt;
import numpy as np;
from setup import encodeLabel, decodeLabel;
import random;

class Model:
    resnet50: ResNet50 = None;
    resnet50V2: ResNet50V2 = None;
    resnet50V2_regressor: ResNet50V2 = None;
    resnet50V2_combined: ResNet50V2 = None;

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

    def train(self, model, X_train: np.ndarray, y_train: np.ndarray | tuple, X_val: np.ndarray, y_val: np.ndarray | tuple, modelName: str, isRegression: bool = False, isRegularized: bool = False):
        '''
        Train a specific classifier.
        '''
        if (isRegression != True):
            # Encode the labels only if it is a classification problem
            y_train_encoded = encodeLabel(y_train);
            y_val_encoded = encodeLabel(y_val);
            print(f"Unique Encoded Labels in training set: {np.unique(y_train_encoded)}");
            print(f"Unique Encoded Labels in validation set: {np.unique(y_val_encoded)}");
        else:
            if (isRegularized == True):

                # Extract labels from classification and regression problems
                y_train_class = y_train[0];
                y_train_reg = y_train[1];
                y_val_class = y_val[0];
                y_val_reg = y_val[1];

                # Perform One Hot Encoding to ensure 4 categories are remained before training
                y_train_class = to_categorical(encodeLabel(y_train_class), num_classes=constants.NUM_CLASSES);
                y_val_class = to_categorical(encodeLabel(y_val_class), num_classes = constants.NUM_CLASSES);

                # Ensure the shapes of the labels in classification problem aligns with the combined model
                if (y_train_class.ndim == 3 and y_train_class.shape[1] == 4):
                    y_train_class = y_train_class.reshape(-1, 9);
                if (y_val_class.ndim == 3 and y_val_class.shape[1] == 4):
                    y_val_class = y_val_class.reshape(-1, 9); 
                
                print(f"Unique Encoded Labels for Classification in training set: {np.unique(y_train_class)}");
                print(f"Unique Encoded Labels for Classification in validation set: {np.unique(y_val_class)}");

        # Train
        if (isRegularized == True):
            history = model.fit(
                X_train, 
                {'classification_output': y_train_class, 'regression_output': y_train_reg}, 
                epochs=constants.epoch, 
                validation_data=(
                    X_val, 
                    {'classification_output': y_val_class, 'regression_output': y_val_reg}
                )
            );
        else:
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
                if (isRegularized != True):
                    model.save("resnet50V2-regression-posttrained.h5");
                    self.resnet50V2_regressor = model;
                else:
                    model.save("resnet50V2-combined-posttrained.h5");
                    self.resnet50V2_combined = model;
            else:
                model.save("resnet50V2-posttrained.h5");
                self.resnet50V2 = model;

        self.plotLearningCurve(history, isRegularized);
        return history;

    def plotLearningCurve(self, history, isRegularized: bool = False):
        '''
        Plot training & validation accuracy values
        '''
        # deduce the string of the metric to be used
        k = '';
        if (isRegularized == True):
            metrics = list(history.history.keys());
            for metric in metrics:
                plt.figure();
                plt.plot(history.history[metric], label=f'Train {metric}');
                val_metric = f'val_{metric}';
                if val_metric in metrics:
                    plt.plot(history.history[val_metric], label=f'Validation {metric}');
                # Format the plot
                plt.xlabel('Epochs');
                plt.ylabel(metric.replace('_', ' ').capitalize());
                plt.title(f'{metric.replace("_", " ").capitalize()} over Epochs');
                plt.legend();
                plt.show();
            return;
        else:
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

    def evaluate(self, model, X_train: np.ndarray, y_train: np.ndarray | tuple, X_val: np.ndarray, y_val: np.ndarray | tuple, X_test: np.ndarray, y_test: np.ndarray | tuple, isRegularized: bool = False):
        '''
        Calculate binary cross-entropy error on each dataset
        '''
        if (isRegularized == True):
            # Extract the labels in each set from classification and regression problems
            y_train_class = y_train[0];
            y_train_reg = y_train[1];
            y_val_class = y_val[0];
            y_val_reg = y_val[1];
            y_test_class = y_test[0];
            y_test_reg = y_test[1];

            # Perform One Hot Encoding to ensure 4 categories are remained before evaluating
            y_train_class = to_categorical(encodeLabel(y_train_class), num_classes=constants.NUM_CLASSES);
            y_val_class = to_categorical(encodeLabel(y_val_class), num_classes = constants.NUM_CLASSES);
            y_test_class = to_categorical(encodeLabel(y_test_class), num_classes = constants.NUM_CLASSES);
            
            # Ensure the shapes of the labels in classification problem aligns with the combined model
            if (y_train_class.ndim == 3 and y_train_class.shape[1] == 4):
                y_train_class = y_train_class.reshape(-1, 9);
            if (y_val_class.ndim == 3 and y_val_class.shape[1] == 4):
                y_val_class = y_val_class.reshape(-1, 9); 
            if (y_test_class.ndim == 3 and y_test_class.shape[1] == 4):
                y_test_class = y_test_class.reshape(-1, 9); 
                
            print(f"Unique Encoded Labels for Classification in training set: {np.unique(y_train_class)}");
            print(f"Unique Encoded Labels for Classification in validation set: {np.unique(y_val_class)}");
            print(f"Unique Encoded Labels for Classification in test set: {np.unique(y_test_class)}");
            
            y_train = {'classification_output': y_train_class, 'regression_output': y_train_reg};
            y_val = {'classification_output': y_val_class, 'regression_output': y_val_reg};
            y_test = {'classification_output': y_test_class, 'regression_output': y_test_reg};
            train_results = model.evaluate(X_train, y_train, verbose=0);
            val_results = model.evaluate(X_val, y_val, verbose=0);
            test_results = model.evaluate(X_test, y_test, verbose=0);

            # extract the cross entropy error
            train_loss = train_results[0];
            val_loss = val_results[0];
            test_loss = test_results[0];
            print(f"Training Binary Cross-Entropy error: {train_loss}");
            print(f"Validation Binary Cross-Entropy error: {val_loss}");
            print(f"Testing Binary Cross-Entropy error: {test_loss}");

            return (train_results, val_results, test_results);
        else:
            train_loss, train_customMetric = model.evaluate(X_train, y_train, verbose=0);
            val_loss, val_customMetric = model.evaluate(X_val, y_val, verbose=0);
            test_loss, test_customMetric = model.evaluate(X_val, y_val, verbose=0);
            print(f"Training Binary Cross-Entropy error: {train_loss}");
            print(f"Validation Binary Cross-Entropy error: {val_loss}");
            print(f"Testing Binary Cross-Entropy error: {test_loss}");
            return ([train_loss, train_customMetric], [val_loss, val_customMetric], [test_loss, test_customMetric]);

    def predict(self, model, X_test: np.ndarray, isRegression: bool = False, isRegularized: bool = False):
        y_pred = model.predict(X_test);
        print(f"Number of Predictions made: {len(y_pred)}");
        if (isRegression != True):
            # Classification
            print(f"Unique Labels in Prediction: {np.unique(decodeLabel(y_pred))}");
        else:
            if (isRegularized == True):
                # Classification
                y_pred_class = y_pred[0];
                # Regression
                y_pred_reg = y_pred[1];

                uniquePred = set();
                for prediction in y_pred_class:
                    prediction = np.unique(decodeLabel(prediction))[0];
                    uniquePred.add(prediction);
                print(f"Unique Labels in Classification Prediction: {list(uniquePred)}");
                print(f"Number of Unique Labels in Regression Prediction: {len(np.unique(y_pred_reg))}");
                print(f"A Regression Prediction looks like: {y_pred_reg[random.randint(0, len(y_pred_reg)-1)]}");
                
                return (y_pred_class, y_pred_reg);
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

    def regularize(self, model):
        '''
        Perform Multi-Task Regularization to overcome the overfitting issue. 
        '''
        inputs = Input(shape=constants.input_shape);

        x = model(inputs);
        if (len(model.output_shape) > 2):
            # Reduce dimensionality if its output is more than 2 dimensions
            x = layers.GlobalAveragePooling2D()(x);

        # Classification Head
        x_classification = layers.Dense(512, activation='relu')(x);
        x_classification = layers.Dense(256, activation='relu')(x_classification);
        classification_output = layers.Dense(constants.NUM_CLASSES, activation='softmax', name='classification_output')(x_classification);
        
        # Regressor Head
        x_regression = layers.Dense(512, activation='relu')(x);
        x_regression = layers.Dense(256, activation='relu')(x_regression);
        COORDINATES_SIZE = 4;
        regression_output = layers.Dense(COORDINATES_SIZE, activation='linear', name='regression_output')(x_regression);
        
        # Combine Models
        newModel = tf.keras.Model(inputs=inputs, outputs=[classification_output, regression_output]);

        # Compile the Model
        newModel.compile(optimizer=constants.optimizer, loss=constants.loss, metrics=constants.combined_metrics);

        # Save the new Model
        self.resnet50V2_combined = newModel;
        newModel.save("resnet50V2_combined_pretrained.h5");

        return newModel;