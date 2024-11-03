import os;
import constants;
import numpy as np;
import skimage.io as imshow;
from skimage.io import imread;
from skimage import exposure;
from skimage.exposure import rescale_intensity;
from skimage.transform import resize;
from skimage.color import rgb2gray;
from skimage.filters import gaussian, sobel;
import matplotlib.pyplot as plt;
import random;
from sklearn.model_selection import train_test_split;
import traceback;
import tensorflow as tf;
from tensorflow.keras.applications.resnet import preprocess_input;
from sklearn.preprocessing import LabelEncoder;

LABEL_ENCODER = LabelEncoder();

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
    # Opens a new textfile (to store image file paths) in "Overwrite" mode
    file_segmentation_paths = open(f"segmentation-{setName}-image-paths.txt", "w");

    X = np.empty(constants.input_shape);
    img_paths, y  = load_image_path(dataset=dataset, setName=setName);

    imgIndToDisplay = random.randint(0, len(img_paths) - 1);
    images = [];
    for imgInd in range(len(img_paths)):
        path = img_paths[imgInd];
        if (dataset == constants.DATASETS_ROOT_DIR[1]):
            # store image paths into a text file
            file_segmentation_paths.write(path);
        if (os.path.isfile(path)):
            # keep in RGB 3 channels
            img_array = imread(path, as_gray=False);

        if (imgInd == imgIndToDisplay):
            displayImage(img_array, y[imgInd], isBefore=True);

        img_array = preprocess(img_array);

        if (imgInd == imgIndToDisplay):
            displayImage(img_array, y[imgInd], isBefore=False);

        # save an image to samples array
        images.append(img_array);

    X = np.array(images);
    file_segmentation_paths.close();
    return X, y;

def encodeLabel(y: np.ndarray):
    '''
    Encode the categorical labels
    '''
    return LABEL_ENCODER.fit_transform(y.reshape(-1, 1));

def decodeLabel(y: np.ndarray):
    '''
    Decode the numeric labels
    '''
    return LABEL_ENCODER.inverse_transform(y.astype(int));

def preprocess(img_array: np.ndarray):
    # Resizing to make the image smaller in resolution
    img_array = setResolution(img_array);

    # Normalize to [0,1] range
    img_array = normalize(img_array);

    img_array = preprocess_input(img_array);

    # histogram equalization - Adaptive equalization
    img_array = exposure.equalize_hist(img_array);

    # sobel edge filter: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
    # for edge detection
    img_array = sobel(img_array);

    # Gaussian blur: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    # for smoothing
    img_array = gaussian(img_array, sigma=constants.gaussianSigma); # sigma = standard deviation for the kernel

    return img_array;

def normalize(img_array: np.ndarray):
    '''
    Normalize an image array to [0, 1] range,
    according to resnet's model standard.
    '''
    img_array = img_array * 255;
    img_array = img_array.astype('uint8');
    img_array = rescale_intensity(img_array, in_range=(0, 255));
    img_array = np.clip(img_array, 0, None);
    return img_array;

def setResolution(img_array: np.ndarray):
    return resize(img_array, constants.input_shape ); # keep original dimensions

def displayImage(img_array: np.ndarray, category: str, isBefore):
    if (isBefore == True):
        plt.title(f"Sample Image in {category.upper()} category BEFORE being processed");
        plt.imshow(img_array);
    else:
        plt.title(f"Sample Image in {category.upper()} category AFTER being processed");
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
    return X_train, X_validation, y_train, y_validation;

# filename in .npy extension
def cacheData(data: np.ndarray, filename: str):
    try:
        np.save(filename, data);
        return True;
    except Exception:
        print(traceback.format_exc())
        return False;

def toGrayscale(img_array: np.ndarray):
    return rgb2gray(img_array);

def convertImages2Grayscale(images: np.ndarray):
    new_images_array = [];
    for img_array in images:
        new_images_array.append(toGrayscale(img_array));
    new_images_array = np.asarray(new_images_array);
    return new_images_array;

def getBoundingBoxCoordinates(binary_image: np.ndarray, toDisplay: bool = False):
    '''
    Get all the bounding box coordinates from a BINARY image.
    '''
    # Find indices of non-zero pixels
    rows, cols = np.where(binary_image[:, :] > 0);
    height, width = binary_image.shape[:2];
    
    if (toDisplay == True):
        # Print the rows and cols (as numpy arrays) of a random image for inspection
        print(f"Number of non-zero row indices in an image in this set: {len(rows)}");
        print(f"Number of non-zero column indices in an image in this set: {len(cols)}");
    
    return {
        # normalize the coordinates
        "height": height,
        "width": width,
        "lower_left_x": min(cols) / width,
        "lower_left_y": min(rows) / height,
        "upper_right_x": max(cols) / width,
        "upper_right_y": max(rows) / height
    };

# Note: the orders of images in all arrays should be the exact same.
def saveBBoxCoordinates(images_set: np.ndarray, y_images_set: np.ndarray, img_paths: str, setName: str):
    # Prerequisite: Open a file in "Append" mode
    if (os.path.exists(constants.BBOX_DIRECTORY) != True):
        os.mkdir(constants.BBOX_DIRECTORY);
    bbox_file = open(constants.BBOX_FILENAME, mode="a");

    # Step 1: parsing the image's name from the corresponding file paths
    img_paths:list = img_paths.split(".\segmentation");
    # removing unnecessary items from the list
    img_paths.remove("");
    image_names = [getImageNameFromPath(img_path) for img_path in img_paths];
    
    # Step 2: Convert all images to binary in grayscale
    images_set_binary = convertImages2Grayscale(images_set);

    # Step 2.5 (optional): Show a random image after processing (to grayscale)
    randInd: int = random.randint(0, len(images_set_binary) - 1);
    randImg: np.ndarray = images_set_binary[randInd];
    print(f"-----{setName}-----");
    displayImage(
        img_array=randImg,
        category=y_images_set[randInd],
        isBefore=False
    );
    print(f"Shape of an image in this set: {randImg.shape}");

    # Step 3: Get all Bounding Box Coordinates
    bbox_items: list = [];
    for ind in range(len(images_set_binary)):
        bbox_dict: dict = {};
        binary_image: np.ndarray = images_set_binary[ind];
        image_name = image_names[ind];

        if (ind == randInd):
            # Print image name
            print(f"Image Name: {image_name}");
        
        bbox = getBoundingBoxCoordinates(
            binary_image=binary_image,
            toDisplay=ind == randInd
        );
        # record all the necessary info to be placed in another file
        height = bbox['height'];
        width = bbox['width'];
        lower_left_x = bbox['lower_left_x'];
        lower_left_y = bbox['lower_left_y'];
        upper_right_x = bbox['upper_right_x'];
        upper_right_y = bbox['upper_right_y'];
    
        if (ind == randInd):
            # print inspection messages
            print(f"Height of an image in {setName}: {height}");
            print(f"Width of an image in {setName}: {width}");
            print(f"Lower left corner of an image in {setName}: ({lower_left_x}, {lower_left_y})");
            print(f"Upper right corner of an image in {setName}: ({upper_right_x}, {upper_right_y})");

        # Save a line to the file
        try:
            line: str = f"{image_name}\t{lower_left_x}\t{lower_left_y}\t{upper_right_x}\t{upper_right_y}\n";
            bbox_file.write(line);
        except:
            print(f"Error printing info of image {image_name}");
    
        # Save to the dictionary
        bbox_dict.update({
            "image_name": image_name,
            "height": height,
            "width": width,
            "lower_left_x": lower_left_x,
            "lower_left_y": lower_left_y,
            "upper_right_x": upper_right_x,
            "upper_right_y": upper_right_y
        });
    
        # Append dictionary to list
        bbox_items.append(bbox_dict);

    # Step 4: Close the file
    bbox_file.close();
    return bbox_items;

def getImageNameFromPath(img_path: str):
    img_path = img_path.lstrip(".\segmentation\set0");
    for allowed_format in constants.ALLOWED_IMAGE_FORMATS:
        img_path = img_path.rstrip(allowed_format);
    img_path = img_path.split('\\');
    image_name = img_path[-1].split(".")[0];
    return image_name;