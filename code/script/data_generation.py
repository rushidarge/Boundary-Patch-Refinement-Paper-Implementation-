import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon 
import json
import os
from PIL import Image,ImageDraw
import cv2
from tqdm import tqdm
import time
import json
import datetime
# Opening JSON file
f = open('data_creation.json')
data_config = json.load(f)

files_path = data_config['data_path']

creating_data_files = os.listdir(files_path)
creating_data_files = [each_file for each_file in creating_data_files if each_file.endswith('jpg') or each_file.endswith('json') or each_file.endswith('JPG')]

all_images_files = [each_file for each_file in creating_data_files if each_file.endswith(('jpg','JPG'))]
print("### ### "*8)
print("We have total {} images".format(len(all_images_files)))

patch_size = data_config['patch_size']
right = patch_size//2
left = patch_size//2
top = patch_size//2
bottom = patch_size//2
patch_tile_size = int(patch_size/2)

done_img = []
binary_single_part_mask_path = data_config['binary_single_part_mask_path']
for each_og_img in tqdm(all_images_files):
    current_file_name = each_og_img.split('.')[0]
    current_file_ext = each_og_img.split('.')[-1]
    if current_file_name not in done_img:
        # print(current_file_name)
        current_img_read = Image.open(files_path+current_file_name+'.'+current_file_ext)
        reading_current_json_file = open(files_path+current_file_name+'.json')
        current_img_json = json.load(reading_current_json_file)
        width, height = current_img_read.size
        new_width = width + right + left
        new_height = height + top + bottom
        current_image_padded = Image.new(current_img_read.mode, (new_width, new_height), (0, 0, 0))
        current_image_padded.paste(current_img_read, (left, top))
        
        for mask_id in current_img_json["shapes"]:
            poly_points=mask_id["points"]
            add_pad_to_point=[]
            for org_point in poly_points:
                add_pad_to_point.append([org_point[0]+patch_tile_size,org_point[1]+patch_tile_size])
            points = np.array(add_pad_to_point)
            
            single_part_mask_image=np.zeros(np.array(current_image_padded).shape,dtype=np.int32)
            cv2.fillPoly(single_part_mask_image, pts=np.int32([points]), color=(255,255,255))
            current_binary_single_part_mask = "{}#{}#{}.jpg".format(binary_single_part_mask_path, mask_id["label"], current_file_name)
            # save the image
            cv2.imwrite(current_binary_single_part_mask,single_part_mask_image)
            current_single_part_mask_image_read =Image.open(current_binary_single_part_mask)

            for point_id in range(len(poly_points)):
                current_p = poly_points[point_id]
                current_padded_point = [current_p[0]+patch_tile_size , current_p[1]+patch_tile_size]
                og_img_cropped_path = data_config['image_patch'] + "{}#{}#{}.jpg".format(mask_id["label"], point_id, current_file_name)
                masked_img_cropped_path = data_config['mask_patch'] + "{}#{}#{}.jpg".format(mask_id["label"], point_id, current_file_name)
               
                og_img_cropped=current_image_padded.crop((int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[0]-32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[1]-32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[0]+32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[1]+32)))
                og_img_cropped.save(og_img_cropped_path)
                # single_part_mask_image_pillow = Image.fromarray(np.uint8(single_part_mask_image)).convert('RGB')
                masked_img_cropped=current_single_part_mask_image_read.crop((int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[0]-32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[1]-32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[0]+32), int((int(current_padded_point[0]-32), int(current_padded_point[1]-32), int(current_padded_point[0]+32), int(current_padded_point[1]+32))[1]+32)))
                masked_img_cropped.save(masked_img_cropped_path)
    done_img.append(current_file_name)
print("***   ***   "*10)
print("We complete the {} of images out of {}".format(len(done_img), len(all_images_files)))
print("Complete ratio is {}".format(len(done_img)/len(all_images_files)))
print("***   ***   "*10)

binary_patch = os.listdir(data_config['mask_patch'])
img_patch = os.listdir(data_config['image_patch'])

df = pd.DataFrame([binary_patch, img_patch]).T
df.columns = ['binary_patch', 'img_patch']
df['binary_patch_path'] = data_config['mask_patch'] + df['binary_patch']
df['img_patch_path'] = data_config['image_patch'] + df['img_patch']
print(df.shape)
print(df.head(3))

# get current time and date add that into name of csv file
current_dt = datetime.now()
current_dt = current_dt.strftime("%d-%m-%Y_%H-%M-%S")
csv_file_name = data_config['path_to_save_csv']+'patch_data_{}.csv'.format(current_dt)
df.to_csv(csv_file_name, index=False)
print("Your csv save at {}".format(data_config['path_to_save_csv']))