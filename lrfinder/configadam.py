import os

# initialize the list of class label names
CLASSES = ["top", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-5
MAX_LR = 1e-3
BATCH_SIZE = 64
STEP_SIZE = 6
CLR_METHOD = "triangular"
NUM_EPOCHS = 50

# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["lrfinder/output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["lrfinder/output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["lrfinder/output", "clr_plot.png"])