import os;

# datasets
STONEFLY = "stonefly";
SEGMENTATION = "segmentation";

# categories
CAL = 'cal';
DOR = 'dor';
HES = 'hes';
ISO = 'iso';
MOS = 'mos';
PTE = 'pte';
SWE = 'swe';
YOR = 'yor';
ZAP = 'zap';

# sets
set0 = 'set0';
set1 = 'set1';

LABELS = [CAL, DOR, HES, ISO, MOS, PTE, SWE, YOR, ZAP];
ALLOWED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png');

# paths
PROJECT_ROOT_DIR = ".";
DATASETS_ROOT_DIR = ['stonefly', 'segmentation'];
BBOX_DIRECTORY = "stoneflies";
BBOX_FILENAME = os.path.join(BBOX_DIRECTORY, "bbox.txt");

VALIDATION_SIZE = 0.2; # a ratio of split
random_state = 42;

# preprocessing parameters
img_height, img_width = 96, 128;
gamma = 0.8;
gaussianSigma = 0.5;  

# Models
RESNET50 = "ResNet50";
RESNET50V2 = "ResNet50V2";

input_shape = (img_height, img_width, 3);
optimizer = 'adam';
loss = 'binary_crossentropy';
metrics = ['accuracy'];
epoch = 10;