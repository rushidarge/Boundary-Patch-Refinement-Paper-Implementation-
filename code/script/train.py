
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coef, iou
from resnet50_unet import build_resnet50_unet
from tensorflow.keras.utils import CustomObjectScope

import json
# Opening JSON file
f = open('train_config.json')
train_config = json.load(f)

print('#### ####'*5)
print(train_config['experiment_name'])

import tensorflow as tf
LIMIT = train_config["GPU_parameters"]  # GPU memory in MB
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)],
)

""" Global parameters """
H = train_config['model_params']['image_size']['height']
W = train_config['model_params']['image_size']['width']

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "Patches", "*.jpg")))
    masks1 = sorted(glob(os.path.join(path, "Binary_Mask", "*.jpg")))
#     masks2 = sorted(glob(os.path.join(path, "ManualMask", "rightMask", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y1, valid_y1 = train_test_split(masks1, test_size=split_size, random_state=42)
#     train_y2, valid_y2 = train_test_split(masks2, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y1, test_y1 = train_test_split(train_y1, test_size=split_size, random_state=42)
#     train_y2, test_y2 = train_test_split(train_y2, test_size=split_size, random_state=42)

    return (train_x, train_y1), (valid_x, valid_y1), (test_x, test_y1)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path1):
    x1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
#     x2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    x = x1 
    x = cv2.resize(x, (W, H))
    x = x/np.max(x)
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y1,):
    def _parse(x, y1):
        x = x.decode()
        y1 = y1.decode()
#         y2 = y2.decode()

        x = read_image(x)
        y = read_mask(y1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y1], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y1, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y1))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(4)
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = train_config['model_params']['batch_size']
    lr = train_config['model_params']['lr_rate']
    num_epochs = train_config['model_params']['num_of_epochs']

    # check train_config['experiment_name'] is there or not if not then create a new directory
    if not os.path.exists(train_config['experiment_name']):
        # create a new directory
        os.makedirs(train_config['experiment_name'])
    model_path = os.path.join(train_config['experiment_name'], "{}_model.h5".format(train_config['experiment_name']))
    # check csv file is there or not if not then create a empty csv file
    if not os.path.exists(os.path.join(train_config['experiment_name'], "model_stats.csv")):
        with open(os.path.join(train_config['experiment_name'], "model_stats.csv"), "w") as f:
            f.write("")
        
        csv_path = os.path.join(train_config['experiment_name'], "model_stats.csv")
    else:
        csv_path = os.path.join(train_config['experiment_name'], "model_stats.csv")
    """ Dataset """
    dataset_path = train_config['dataset_path']
    (train_x, train_y1), (valid_x, valid_y1), (test_x, test_y1) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y1)} ")
    print(f"Valid: {len(valid_x)} - {len(valid_y1)} ")
    print(f"Test: {len(test_x)} - {len(test_y1)} ")

    train_dataset = tf_dataset(train_x, train_y1, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y1, batch=batch_size)

    """ Model """
    if train_config['model_train_new'] == True:
        model = build_resnet50_unet((H, W, 3))
        metrics = [dice_coef, iou, Recall(), Precision()]
        model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    else:
        print('#### ####'*3)
        print('Loading Pretrain Model')
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(train_config['pretrain_model_path'], compile=False)
            metrics = [dice_coef, iou, Recall(), Precision()]
            model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    model.summary()
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path)
    ]

    model.fit(
        train_dataset,
        epochs=train_config['model_params']['num_of_epochs'],
        validation_data=valid_dataset,
        callbacks=callbacks
    )
