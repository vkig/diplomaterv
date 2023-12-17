import os.path
from tqdm import tqdm
import yaml
from config import *


def write_annotation_to_file(annotation):
    annotation_payload = annotation["annotation"]
    if not os.path.exists(os.path.join(ANNOTATION_PATH, str(annotation["points_file"]))):
        with open(os.path.join(ANNOTATION_PATH, str(annotation["points_file"])), 'w') as f:
            f.write("label;x;y;z;length;width;height\n")
    with open(os.path.join(ANNOTATION_PATH, str(annotation["points_file"])), 'a') as f:
        f.write(f'{annotation_payload["label"]};{annotation_payload["translation"]["x"]};{annotation_payload["translation"]["y"]};{annotation_payload["translation"]["z"]};{annotation_payload["box"]["length"]};{annotation_payload["box"]["width"]};{annotation_payload["box"]["height"]}\n')


for filename in FILENAMES:
    annotations = yaml.safe_load(open(os.path.join(ANNOTATION_FILES_PATH, filename + ".yaml"), 'r'))
    for annotation in tqdm(annotations["annotations"], total=len(annotations["annotations"])):
        write_annotation_to_file(annotation)
