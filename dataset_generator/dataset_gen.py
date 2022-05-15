from glob import glob
import os
import glob
from unicodedata import name
import numpy as np
import collections

# Find all the folders with name "Annotations" recursively
def find_all_folders(path):
    annotations_folders = []
    images_folder = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d == "Annotations":
                annotations_folders.append(root)
            if d == "JPEGImages":
                images_folder.append(root)
    return annotations_folders, images_folder


# Finall xml files in the folder
def find_all_xml_jpg_files(path):
    xml_files = []
    jpg_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".xml"):
                xml_files.append(os.path.join(root, f))
            if f.endswith(".jpg"):
                jpg_files.append(os.path.join(root, f))
    return xml_files, jpg_files


if __name__ == "__main__":
    # Find all the folders with name "Annotations" recursively
    annotations_folders, images_folder = find_all_folders("/home/student/Downloads/dataset")
    cleaned_dataset = "/home/student/Downloads/clean_dataset"
    cleaned_dataset_images = "/home/student/Downloads/clean_dataset/JPEGImages"
    cleaned_dataset_annotations = "/home/student/Downloads/clean_dataset/Annotations"
    # create a folder for cleaned dataset, and if it exists delete it
    if os.path.exists(cleaned_dataset):
        os.system("rm -rf " + cleaned_dataset)
    # else:
    os.system(f"mkdir {cleaned_dataset}")
    os.system(f"mkdir {cleaned_dataset_images}")
    os.system(f"mkdir {cleaned_dataset_annotations}")
    all_xml_files = []
    all_jpg_files = []
    for folder in annotations_folders:
        xml_files, jpg_files = find_all_xml_jpg_files(folder)
        all_xml_files.extend(xml_files)
    for folder in images_folder:
        xml_files, jpg_files = find_all_xml_jpg_files(folder)
        all_jpg_files.extend(jpg_files)
    # Print Duplicates in the all_jpf_files list
    print("Duplicates in the all_jpg_files list:")
    just_file_names = [os.path.basename(x) for x in all_jpg_files]
    duplicate_image_list = [item for item, count in collections.Counter(just_file_names).items() if count > 1]
    print("Duplicates:")
    for each_image in all_jpg_files:
        for each_duplicate in duplicate_image_list:
            if each_duplicate in each_image:
                print(each_image)
    print("Starting dataset generation...")
    just_file_names = [x[:-4] for x in just_file_names]
    for i, each_annotation in enumerate([os.path.basename(x) for x in all_xml_files]):
        if each_annotation[:-4] in just_file_names:
            try:
                # get index of the file name in the all_jpg_files list
                index = just_file_names.index(each_annotation[:-4])
                # get the corresponding file name in the all_jpg_files list
                corresponding_image_name = all_jpg_files[index]
                # get the corresponding file name in the all_xml_files list
                corresponding_xml_name = all_xml_files[i]
                # copy the corresponding image file to the cleaned dataset
                os.system(f'cp "{corresponding_image_name}" "{cleaned_dataset_images}"')
                # copy the corresponding xml file to the cleaned dataset
                os.system(f'cp "{corresponding_xml_name}" "{cleaned_dataset_annotations}"')
            except:
                print(f"This annotation file is not in the jpg files list: {each_annotation}")
    # create extra folders according to the VOC
    os.system(f"mkdir -p {cleaned_dataset}/ImageSets/Main")
    # create labels.txt file
    os.system(f"echo 'Goal\nSoccer Ball' > {cleaned_dataset}/labels.txt")
    # read all jpg files in the cleaned dataset
    jpg_files = glob.glob(f"{cleaned_dataset_images}/*.jpg")
    # divide them train test and validation
    # jumble the list
    np.random.shuffle(jpg_files)
    train_jpg_files = jpg_files[: int(len(jpg_files) * 0.8)]
    test_jpg_files = jpg_files[int(len(jpg_files) * 0.8) : int(len(jpg_files) * 0.9)]
    val_jpg_files = jpg_files[int(len(jpg_files) * 0.9) :]
    # create train.txt file
    with open(f"{cleaned_dataset}/ImageSets/Main/train.txt", "w+") as f:
        for each_file in train_jpg_files:
            f.write(f"{os.path.basename(each_file)[:-4]}\n")
    # create test.txt file
    with open(f"{cleaned_dataset}/ImageSets/Main/test.txt", "w+") as f:
        for each_file in test_jpg_files:
            f.write(f"{os.path.basename(each_file)[:-4]}\n")
    # create val.txt file
    with open(f"{cleaned_dataset}/ImageSets/Main/val.txt", "w+") as f:
        for each_file in val_jpg_files:
            f.write(f"{os.path.basename(each_file)[:-4]}\n")
    # create trainval.txt file
    with open(f"{cleaned_dataset}/ImageSets/Main/trainval.txt", "w+") as f:
        for each_file in train_jpg_files:
            f.write(f"{os.path.basename(each_file)[:-4]}\n")
        for each_file in val_jpg_files:
            f.write(f"{os.path.basename(each_file)[:-4]}\n")
    print("Done")
