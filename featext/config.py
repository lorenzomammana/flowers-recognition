# import the necessary packages
import os

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = '/home/ubuntu/flower-classification/flower_data_original'

# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "valid"

# initialize the list of class label names
CLASSES = 102

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 2
CLR_METHOD = "triangular"
NUM_EPOCHS = 50

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot_"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot_"])
