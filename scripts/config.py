FILTERED_DATA_PATH = "/home/vkig/diplomaterv/filtered_data"
ANNOTATION_FILES_PATH = "/home/vkig/diplomaterv/database/m_hid"
ANNOTATION_PATH = "/home/vkig/diplomaterv/database/annotations"
POINTS_PATH = "/home/vkig/diplomaterv/database/points"

BACKGROUND_MODEL_TOPIC = "/background_model"
BOUNDING_BOX_TOPIC = "/bounding_box"
FILTERED_CLOUD_TOPIC = "/filtered_cloud"
CAMERA_TOPIC = "/camera/image_color"

ANNOTATION_OUTPUT_FILE = "/home/vkig/diplomaterv/database/annotations.yaml"

FILENAMES = ["m_hid_1_1", "m_hid_1_2", "m_hid_2_1", "m_hid_2_2", "m_hid_3_1", "m_hid_3_2", "m_hid_3_3", "m_hid_3_7",
             "p4_11_52"]

POINTS_IN_THE_BOX_THRESHOLD = 3
TIME_THRESHOLD = 10000000

X_MAX = 120
X_MIN = 0
Y_MAX = 30
Y_MIN = -20
Z_MAX = 15
Z_MIN = -6

B = 6
C = 2
O = 1
N = 16

CLUSTERING_EPS = 0.75
BATCH_SIZE = 32

INPUT_SHAPE = (4, 128)
OUTPUT_SHAPE = (N, B + O + C)
SAVE_PATH = "/home/vkig/diplomaterv/model_saves"
ACCURACY_FILE = "/home/vkig/diplomaterv/accuracies.txt"
LOSS_FILES = "/home/vkig/diplomaterv/losses"
FRAME_NAME = "livox_frame"

INDEX_PATH = "/home/vkig/diplomaterv/database/indices"
TEST_INDEX_FILE = "/home/vkig/diplomaterv/database/indices/test_indices.csv"
TRAIN_INDEX_FILE = "/home/vkig/diplomaterv/database/indices/train_indices.csv"
VALIDATION_INDEX_FILE = "/home/vkig/diplomaterv/database/indices/validation_indices.csv"

PROCESSED_DATA_TOPIC = "/processed_data"
BOUNDING_BOXES_TOPIC = "/objects"

AUGMENTED_DATABASE_SIZE = 21536
DATABASE_SIZE = 5219
