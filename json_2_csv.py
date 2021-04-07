"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import csv
from tqdm import tqdm


def json_to_csv(path): 

    classes_names = []
    json_list = []
    df = pd.read_json(path)

    # bb_n = []
    # region_types = []    
    # for region in df['regions']:
    #     for sub in region:
            # bb_n.append(sub['shape_attributes']['name']) # array(['polygon', 'rect'], dtype=object) 
            # if 'type' in sub['region_attributes']:
            #     region_types.append(sub['region_attributes']['type'])
            # array(['e', 'h', 's', 'n', 'a', 'ns', 'H', 'ns1', 'ss', 'n\n'], dtype=object)    
    
    for item in tqdm(range(len(df))):
        flnm = df['filename'][item]
        for dct in df['regions'][item]:
            if 'type' in dct['region_attributes']: 
                if dct['region_attributes']['type']=='s' or dct['region_attributes']['type'] == 'ss': 
                    classes_names.append(dct["shape_attributes"]['name'])
                    value = (
                        flnm,
                        dct["shape_attributes"]['width'],
                        dct["shape_attributes"]['height'],
                        dct["shape_attributes"]['name'], # rect 
                        dct["shape_attributes"]['x'],  
                        dct["shape_attributes"]['y'],
                        dct["shape_attributes"]['x'] + dct["shape_attributes"]['width'], 
                        dct["shape_attributes"]['y'] + dct["shape_attributes"]['height'],
                        )
                    json_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    json_df = pd.DataFrame(json_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return json_df, classes_names
    
def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow JSON-to-CSV converter"
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        help="Path to the file where the input json file is stored",
        type=str,
    )
    parser.add_argument(
        "-o", "--outputFile", help="Name of output .csv file (including path)", type=str
    )
    parser.add_argument(
        "-l",
        "--labelMapDir",
        help="Directory path to save label_map.pbtxt file is specified.",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.inputFile is None:
        args.inputFile = 'train_annotations.json'
    if args.outputFile is None:
        args.outputFile =  "./labels.csv"

    assert os.path.isfile(args.inputFile)
    json_df, classes_names = json_to_csv(args.inputFile)
    json_df.to_csv(args.outputFile, index=None)
    print("Successfully converted json to csv.")
    if args.labelMapDir:
        os.makedirs(args.labelMapDir, exist_ok=True)
        label_map_path = os.path.join(args.labelMapDir, "label_map.pbtxt")
        print("Generate `{}`".format(label_map_path))

        # Create the `label_map.pbtxt` file
        pbtxt_content = ""
        for i, class_name in enumerate(classes_names):
            pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                    i + 1, class_name
                )
            )
        pbtxt_content = pbtxt_content.strip()
        with open(label_map_path, "w") as f:
            f.write(pbtxt_content)


if __name__ == "__main__":
    main()
