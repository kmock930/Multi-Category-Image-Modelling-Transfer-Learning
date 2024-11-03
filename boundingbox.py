import numpy as np;
import os;
import random;
import constants;
from setup import getImageNameFromPath, convertImages2Grayscale, displayImage;
from matplotlib import pyplot as plt, patches;
import traceback;

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
    image_names = [getImageNameFromPath(img_path, setName) for img_path in img_paths];
    
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

def drawBoundingBox(bboxFilepath: str, X_imageSet: np.ndarray, y_imageSet: np.ndarray, setName: str):
    '''
    Draw a bounding box around the object inside an image.\n
    Source: https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
    '''
    try:
        bboxFile = open(bboxFilepath, "r");
        bboxFileContents = bboxFile.readlines();
        randInd: int = random.randint(0, len(bboxFileContents) - 1);
        for currImgInd in range(len(bboxFileContents)):
            imageBBOX = bboxFileContents[currImgInd];
            imageName, lower_left_x, lower_left_y, upper_right_x, upper_right_y = imageBBOX.split('\t');
            lower_left_x = float(lower_left_x);
            lower_left_y = float(lower_left_y);
            upper_right_x = float(upper_right_x);
            upper_right_y = float(upper_right_y);

            # Calculate box dimensions
            box_x = lower_left_x;
            box_y = lower_left_y;
            box_width = upper_right_x - lower_left_x;
            box_height = upper_right_y - lower_left_y;
            
            # Display a random image (for easier debugging)
            if (currImgInd == randInd):
                # Plot the image
                if (currImgInd < len(X_imageSet)):
                    image = X_imageSet[currImgInd];
                else:
                    # to avoid index out of bound
                    image = X_imageSet[-1];
                fig, ax = plt.subplots();
                ax.imshow(image);
            
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (box_x, box_y), box_width, box_height,
                    linewidth=2, edgecolor='r', facecolor='none'
                );
            
                # Add the rectangle to the plot
                ax.add_patch(rect);
            
                plt.show();

        bboxFile.close();
        return True;
    except Exception as e:
        traceback.print_exception(e);
        return False;