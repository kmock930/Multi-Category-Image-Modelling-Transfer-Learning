import os;
import constants;
import numpy as np;
import skimage.io as imshow;
from skimage.io import imread;
from skimage import exposure;
from skimage.transform import resize;
from skimage.color import rgb2gray;
from skimage.filters import gaussian, sobel;
import matplotlib.pyplot as plt;
import random;
from sklearn.model_selection import train_test_split;
import traceback;

def load_image_path(dataset: str, setName: str):
    y = np.array([]); # labels

    imagePaths = np.array([]);

    if (dataset == constants.DATASETS_ROOT_DIR[0]):
        # stonefly
        dataset_path = os.path.join(constants.PROJECT_ROOT_DIR, dataset);
        if (os.path.isdir(dataset_path)):
            for category in os.listdir(dataset_path):
                category_dir  = os.path.join(dataset_path, category);
                if (os.path.isdir(category_dir) == True):
                    for set in os.listdir(category_dir):
                        setPath = os.path.join(category_dir, set);
                        if (os.path.isdir(setPath) and set == setName):
                            for imagePath in os.listdir(setPath):
                                imagePath: str = os.path.join(setPath, imagePath);
                                if (os.path.isfile(imagePath) == True):
                                    if (imagePath.endswith(constants.ALLOWED_IMAGE_FORMATS)):
                                        # add to sample and label arrays
                                        imagePaths = np.append(imagePaths, imagePath);

                                        y = np.append(y, category);
                                else: #still a directory
                                    for output in os.listdir(imagePath):
                                        outputPath = os.path.join(imagePath, output);
                                        if (os.path.isfile(outputPath)):
                                            if (outputPath.endswith(constants.ALLOWED_IMAGE_FORMATS) == True):
                                                # add to sample and label arrays
                                                imagePaths = np.append(imagePaths, imagePath);
                                                y = np.append(y, category);
                                        elif (os.path.isdir(outputPath)):
                                            raise NameError(f"The path {imagePath} contains unexpected subdirectories inside.");
                        else:
                            if (os.path.isfile(setPath) == True):
                                raise NameError(f"The path {categoryPath} contains unexpected files inside.");
                else:
                    raise NameError(f"The path {dataset_path} contains unexpected files inside.");
        else:
            raise NameError(f"The root directory contains unexpected files inside.");
                    
    elif (dataset == constants.DATASETS_ROOT_DIR[1]):
        # segmentation
        dataset_path = os.path.join(constants.PROJECT_ROOT_DIR, dataset);
        if (os.path.isdir(dataset) == True):
            for setName in os.listdir(dataset):
                setPath = os.path.join(dataset_path, setName);
                if (os.path.isdir(setPath)):
                    for category in os.listdir(setPath):
                        categoryPath = os.path.join(setPath, category);
                        if (os.path.isdir(categoryPath)):
                            for imgPath in os.listdir(categoryPath):
                                imgPath = os.path.join(categoryPath, imgPath);
                                if (os.path.isfile(imgPath)):
                                    # add to sample and label arrays
                                    imagePaths = np.append(imagePaths, imgPath);

                                    y = np.append(y, category);
                                else:
                                    raise NameError(f"The path {imgPath} contains unexpected subdirectories inside.");
                        else:
                            raise NameError(f"The path {categoryPath} contains unexpected files inside.");
                else:
                    raise NameError(f"The path {setPath} contains unexpected files inside.");
        else:
            raise NameError(f"The path {dataset_path} contains unexpected files inside.");

    return imagePaths, y;

def load_images(dataset: str, setName: str):
    X = np.empty((0, constants.img_height, constants.img_width));
    img_paths, y  = load_image_path(dataset=dataset, setName=setName);
    imgIndToDisplay = random.randint(0, len(img_paths) - 1);
    for imgInd in range(len(img_paths)):
        if (os.path.isfile(img_paths[imgInd])):
            img_array = imread(img_paths[imgInd]);

        if (imgInd == imgIndToDisplay):
            displayImage(img_array, y[imgInd], isBefore=True);

        img_array = preprocess(img_array);

        img_array = normalize(img_array);

        if (imgInd == imgIndToDisplay):
            displayImage(img_array, y[imgInd], isBefore=False);

        # save an image to samples array
        X = np.append(X, [img_array], axis=0);
    return X, y;

def preprocess(img_array: np.ndarray):
    # Resizing to make the image smaller in resolution
    img_array = setResolution(img_array);

    # Grayscale
    if (img_array.ndim == 3): # currently a color image in RGB
        img_array = rgb2gray(img_array);

    # Normalize to [0,1] range
    img_array = normalize(img_array);

    # point processing - Gamma Correction
    # to adjust the brightness
    img_array = exposure.adjust_gamma(img_array, gamma=constants.gamma);

    # histogram equalization - Adaptive equalization
    img_array = exposure.equalize_hist(img_array);

    # sobel edge filter: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
    # for edge detection
    img_array = sobel(img_array);

    # Gaussian blur: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    # for smooting
    img_array = gaussian(img_array, sigma=constants.gaussianSigma); # sigma = standard deviation for the kernel

    return img_array;

def normalize(img_array: np.ndarray):
    return img_array / 255.0;  # Normalize to [0, 1] range

def setResolution(img_array: np.ndarray):
    return resize(img_array, (constants.img_height, constants.img_width)); # keep original dimensions

def displayImage(img_array: np.ndarray, category: str, isBefore):
    if (isBefore == True):
        plt.title(f"Sample Image in {category} category BEFORE being processed");
        plt.imshow(img_array);
    else:
        plt.title(f"Sample Image in {category} category AFTER being processed");
        plt.imshow(img_array, cmap="gray");
    
    plt.show();

def split(X: np.ndarray, y: np.ndarray):
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y, 
        test_size=constants.VALIDATION_SIZE, 
        shuffle=True,
        stratify=y,
        random_state=constants.random_state
    );
    # caching
    cacheData(data=X_train, filename="X_train.npy");
    cacheData(data=X_validation, filename="X_valdation.npy");
    cacheData(data=y_train, filename="y_-train.npy");
    cacheData(data=y_validation, filename="y_validation.npy");
    return X_train, X_validation, y_train, y_validation;

# filename in .npy extension
def cacheData(data: np.ndarray, filename: str):
    try:
        np.save(filename, data);
        return True;
    except Exception:
        print(traceback.format_exc())
        return False;
