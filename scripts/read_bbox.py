import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
os.mkdir('dataset/open-images/bbox')
dir_list=os.listdir('dataset/open-images/images')
dir_list.sort()
for dir_name in dir_list:
    print(dir_name)
    if 'validation' in dir_name:
        csv_file_path='dataset/open-images/annotations/validation-annotations-bbox.csv'
    elif 'test' in dir_name:
        csv_file_path='dataset/open-images/annotations/test-annotations-bbox.csv'
    else:
        csv_file_path='dataset/open-images/annotations/oidv6-train-annotations-bbox.csv'

    download_dir = os.path.join('dataset/open-images/images', dir_name)
    label_dir = os.path.join('dataset/open-images/bbox', dir_name)
    os.mkdir(label_dir)

    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
    images_label_list = list(set(downloaded_images_list))
    df_val = pd.read_csv(csv_file_path)
    groups = df_val.groupby(df_val.ImageID)
    for image in tqdm(images_label_list):
        try:
            current_image_path = os.path.join(download_dir, image + '.jpg')
            dataset_image = cv2.imread(current_image_path)
            # print(image)
            boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
            boxes_new=[]
            for box in boxes:
                if not((box[1]-box[0])*(box[3]-box[2])>0.8 or (box[1]-box[0])*(box[3]-box[2])<0.02):
                    box[0] *= int(dataset_image.shape[1])
                    box[1] *= int(dataset_image.shape[1])
                    box[2] *= int(dataset_image.shape[0])
                    box[3] *= int(dataset_image.shape[0])
                    boxes_new.append([box[0],box[1],box[2],box[3]])
            
            if len(boxes_new)>0:
                file_name = str(image.split('.')[0]) + '.txt'
                file_path = os.path.join(label_dir, file_name)
                # print(file_path)
                if os.path.isfile(file_path):
                    f = open(file_path, 'a')
                else:
                    f = open(file_path, 'w')

                for box in boxes_new:
                        # each row in a file is name of the class_name, XMin, YMin, XMax, YMax (left top right bottom)
                    print(box[0], box[2], box[1], box[3], file=f)
        except Exception as e:
            pass
