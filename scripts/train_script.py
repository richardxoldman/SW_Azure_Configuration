import argparse
from azureml.core import Run
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from logging import exception
AUTOTUNE = tf.data.experimental.AUTOTUNE
import math

import pathlib
from os.path import exists, join
from os import mkdir, listdir


run = Run.get_context()

parser = argparse.ArgumentParser()

parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point')

parser.add_argument('--initial_learning_rate', type=float, dest='initial_learning_rate')
parser.add_argument('--optimizer', type=str, dest='optimizer')
parser.add_argument('--mobilenet_version', type=str, dest='mobilenet_version')
parser.add_argument('--augmentation_variant', type=int, dest='augmentation_variant')
parser.add_argument('--limit_artificials', type=int, dest='limit_artificials')
args = parser.parse_args()

run.log('initial_learning_rate',  args.initial_learning_rate)
run.log('optimizer',  args.optimizer)
run.log('mobilenet_version',  args.mobilenet_version)
run.log('agumentation_variant',  args.augmentation_variant)
run.log('limit_artificials', args.limit_artificials)


if args.augmentation_variant == 1:
    noise_type = 'gausian'
    ag_strength = 0.05
if args.augmentation_variant == 2:
    noise_type = 'gausian'
    ag_strength = 0.1
if args.augmentation_variant == 3:
    noise_type = 'poisson'
    ag_strength = 0.05
if args.augmentation_variant == 4:
    noise_type = 'poisson'
    ag_strength = 0.1


ag_brigthess = False
ag_contrast = False
ag_saturation = False
ag_hue = False

if noise_type == 'gaussian':
    ag_gaussian_noise = True
    ag_poisson_noise = False
else:
    ag_gaussian_noise = False
    ag_poisson_noise = True

ag_strength = float(ag_strength)

use_mobilenet = True
resolution = 224
use_dropout = True
learning_rate_decay = "CosineDecay"
mobilenet_version = args.mobilenet_version
all_trainable = True

epochs = 100
batch_size = 48

max_leaf_size_cm2 = 70
use_root_of_size = True
with_data_augmentation = True
limit_artificial = args.limit_artificials

if limit_artificial == 2500:
    batch_size = 120
if limit_artificial == 5000:
    batch_size = 120
    
pretrained_weights = "imagenet" 
export_name = "_"

def add_gaussian_noise(img_in, strength = 0.05):
    img = img_in
    strength_now =  tf.random.uniform([], minval=0, maxval=strength, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=strength_now, dtype=tf.float32)
    noise_img = tf.add(img, noise) 
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img


def add_poisson_noise(img_in, strength = 0.05):
    img = img_in
    strength_now =  tf.random.uniform([], minval=0, maxval=strength, dtype=tf.float32)

    noise = tf.random.poisson(
    tf.shape(img),
    1.0,
    dtype=tf.dtypes.float32
    )
    noise = noise - 0.5
    noise = noise * strength_now
    noise_img = tf.add(img, noise) 
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img


def data_augmentation(img, label):
    global export_name
    img_out = img
    if(with_data_augmentation):
        export_name += "_ag"
        if(ag_brigthess):
            img_out = tf.image.random_brightness(img_out, 0.17)
            export_name += "_br"
        if(ag_contrast):
            img_out = tf.image.random_contrast(img_out, 0.72, 1.27)
            export_name += "_ct"
        if(ag_saturation):
            img_out = tf.image.random_saturation(img_out, 0.72, 1.27)
            export_name += "_st"
        if(ag_hue):
            img_out = tf.image.random_hue(img_out, 0.07)
            export_name += "_hu"
        if(ag_gaussian_noise):
            export_name += "_gn"
            img_out = tf.cast(add_gaussian_noise(img_out, ag_strength), tf.float32)
        if(ag_poisson_noise):
            export_name += "_pn"
            img_out = tf.cast(add_poisson_noise(img_out, ag_strength), tf.float32)
        
        # img_out = tf.image.random_jpeg_quality(img_out, 75, 100)
    return (img_out, label)


def ResBlock(inputs, id):
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="res_a"+str(id))(inputs)
    x = layers.Conv2D(64, 3, padding="same", name="res_b"+str(id))(x)
    x = layers.Add()([inputs, x])
    return x


def mae_cm2(y_true, y_pred):
    y_truef = tf.cast(y_true, tf.float32)
    y_predf = tf.cast(y_pred, tf.float32)
    y_truep = tf.math.pow(y_truef, 2)
    y_predp = tf.math.pow(y_predf, 2)

    differenceA = tf.math.abs((y_truep - y_predp))
    meanA = tf.reduce_mean(differenceA, axis=-1) 
    return meanA


def mae_root(y_true, y_pred):
    y_truef = tf.cast(y_true, tf.float32)
    y_predf = tf.cast(y_pred, tf.float32)
    y_truep = tf.math.sqrt(y_truef)
    y_predp = tf.math.sqrt(y_predf)

    differenceA = tf.math.abs((y_truep - y_predp))
    meanA = tf.reduce_mean(differenceA, axis=-1) 
    return meanA


tf.keras.backend.set_epsilon(1)
def mean_absolute_percentage_error_cm(y_true, y_pred):
    y_truef = tf.cast(y_true, tf.float32)
    y_predf = tf.cast(y_pred, tf.float32)
    y_truep = tf.math.pow(y_truef, 2)
    y_predp = tf.math.pow(y_predf, 2)
    diff = tf.math.abs((y_truep - y_predp) / tf.clip_by_value(tf.math.abs(y_truep),tf.keras.backend.epsilon(),1000000))
    return 100. * tf.reduce_mean(diff, axis=-1)


def mean_squere_percentage_error(y_true, y_pred):
    y_truef = tf.cast(y_true, tf.float32)
    y_predf = tf.cast(y_pred, tf.float32)
    # y_truep = tf.math.pow(y_truef, 2)
    # y_predp = tf.math.pow(y_predf, 2)
    diff = tf.math.pow((y_truef - y_predf) / tf.clip_by_value(tf.math.abs(y_truef),tf.keras.backend.epsilon(),1000000), 2)
    return 100. * tf.reduce_mean(diff, axis=-1)
#     return my_loss_fn


metrics=["mse", "mae"]

if(use_root_of_size):
    metrics.append(mae_cm2)
    metrics.append(mean_absolute_percentage_error_cm)
else:
    metrics.append(mae_root)
    metrics.append("mean_absolute_percentage_error")

print("Loading Data...")
data_root = run.input_datasets['training_files']
data_root = pathlib.Path(data_root)  

all_image_paths = list(data_root.glob(join('*', 'ph*')))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths = list(filter(lambda path: int(path.split("/")[-1][2:].split("_")[0]) < max_leaf_size_cm2, all_image_paths))


image_paths_real = list(filter(lambda path: "frame" not in str(path), all_image_paths))
image_paths_21 = list(filter(lambda path: str(path.split("/")[-1]).count('_') < 3, image_paths_real))
image_paths_22 = list(filter(lambda path: str(path.split("/")[-1]).count('_') >= 3, image_paths_real))

all_image_paths = list(filter(lambda path: "frame" in str(path), all_image_paths))
if(len(all_image_paths) > 16):
    random.shuffle(all_image_paths)
    if(len(all_image_paths) > limit_artificial):
        all_image_paths = all_image_paths[:limit_artificial]

image_count = len(all_image_paths)


dirs_names = sorted(item.name for item in data_root.glob(join('*', '*')) if  not item.is_dir())

dirs_names_real = list(filter(lambda path: "frame" not in str(path), dirs_names))
dirs_names_21 = list(filter(lambda path: str(path).count('_') < 3, dirs_names_real))
dirs_names_22 = list(filter(lambda path: str(path).count('_') >= 3, dirs_names_real))

dirs_names = list(filter(lambda path: "frame" in str(path), dirs_names))
if(len(dirs_names) > limit_artificial):
    dirs_names = dirs_names[:limit_artificial]
label_names = []
for lbl in dirs_names:
    if("ph" in lbl and "lys" in lbl):
        label_names.append(float(lbl.split("ph")[1].split("lys")[0].replace("_", ".")))
label_names_21 = []
for lbl in dirs_names_21:
    if("ph" in lbl and "lys" in lbl):
        label_names_21.append(float(lbl.split("ph")[1].split("lys")[0].replace("_", ".")))
label_names_22 = []
for lbl in dirs_names_22:
    if("ph" in lbl and "lys" in lbl):
        label_names_22.append(float(lbl.split("ph")[1].split("lys")[0].replace("_", ".")))

label_to_index = dict((index, name) for index, name in enumerate(label_names))
label_to_index_21 = dict((index, name) for index, name in enumerate(label_names_21))
label_to_index_22 = dict((index, name) for index, name in enumerate(label_names_22))



all_image_labels = [float((pathlib.Path(path).name).split("ph")[1].split("lys")[0].replace("_", "."))
                    for path in all_image_paths]
image_labels_21 = [float((pathlib.Path(path).name).split("ph")[1].split("lys")[0].replace("_", "."))
                    for path in image_paths_21]
image_labels_22 = [float((pathlib.Path(path).name).split("ph")[1].split("lys")[0].replace("_", "."))
                    for path in image_paths_22]


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [resolution, resolution])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def caption_image(path):
    return (pathlib.Path(path).name).split("ph")[1].split("lys")[0].replace("_", ".")


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = ()
label_ds = ()
image_label_ds = None

if(len(path_ds) > 0):
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.float16))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))



path_ds_21 = tf.data.Dataset.from_tensor_slices(image_paths_21)
image_ds_21 = path_ds_21.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds_21 = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels_21, tf.float16))


image_label_ds_21 = tf.data.Dataset.zip((image_ds_21, label_ds_21))

path_ds_22 = tf.data.Dataset.from_tensor_slices(image_paths_22)
image_ds_22 = path_ds_22.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds_22 = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels_22, tf.float16))


image_label_ds_22 = tf.data.Dataset.zip((image_ds_22, label_ds_22))


if use_root_of_size:
    all_image_labels = tf.sqrt(all_image_labels)
    image_labels_21 = tf.sqrt(image_labels_21)
    image_labels_22 = tf.sqrt(image_labels_22)

if use_root_of_size:
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    if(len(path_ds) > 0):
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.float16))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    path_ds_21 = tf.data.Dataset.from_tensor_slices(image_paths_21)
    image_ds_21 = path_ds_21.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    label_ds_21 = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels_21, tf.float16))
    image_label_ds_21 = tf.data.Dataset.zip((image_ds_21, label_ds_21))



    path_ds_22 = tf.data.Dataset.from_tensor_slices(image_paths_22)
    image_ds_22 = path_ds_22.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    label_ds_22 = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels_22, tf.float16))
    image_label_ds_22 = tf.data.Dataset.zip((image_ds_22, label_ds_22))


def get_dataset_partitions_tf(ds, ds_21, ds_22, val_split=0.5, shuffle=True, shuffle_size=10000):
    ds_size = 0
    if(ds != None):
        ds_size = len(ds)
    ds_21_size = len(ds_21)
    ds_22_size = len(ds_22)
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        if(ds != None):
            ds = ds.shuffle(shuffle_size, seed=12)
        ds_21 = ds_21.shuffle(shuffle_size, seed=12)
        ds_22 = ds_22.shuffle(shuffle_size, seed=12)
    
    # train_size = int(train_split * ds_size_test)
    val_size = int(val_split * ds_22_size)
    test_size = int((1.0 - val_split) * ds_22_size)
    
    val_ds = ds_22.take(val_size)    
    test_ds = ds_22.skip(val_size).take(test_size)
    train_ds = ds_21
    if(ds_size > 0):
        train_ds = ds_21.concatenate(ds)    
    if shuffle:
        train_ds = train_ds.shuffle(shuffle_size, seed=12)
        
    print("train: "+str(len(train_ds))+" photos, validation: "+str(len(val_ds))+ " photos, test: "+str(len(test_ds))+" photos")
    
    
    return train_ds, val_ds, test_ds,


train_ds, val_ds, test_ds = get_dataset_partitions_tf(image_label_ds,image_label_ds_21,image_label_ds_22)



train_ds = train_ds.prefetch(buffer_size=int(batch_size*2))
val_ds = val_ds.prefetch(buffer_size=int(batch_size*2))

train_batches = (
    train_ds
        # .cache()
        .shuffle(40)
        .batch(int(batch_size))
        # .repeat()
        .map(lambda img, label: data_augmentation(img, label))
        .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_batches = (
    val_ds
        # .cache()
        .shuffle(40)
        .batch(int(batch_size))
        # .repeat()
        # .map(Augment())
        # .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_ds.batch(int(batch_size))


def make_all_trainable(model):
    for layer in model.layers:
        layer.trainable = True
        try:
            make_all_trainable(layer)
        except:
            errrr = True


def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    if(mobilenet_version == "Small"):
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            input_shape=input_shape,
            weights=pretrained_weights,
            classes=1000,
            include_preprocessing=False
            )
        base_model.trainable = True
        make_all_trainable(base_model)
        x = base_model(inputs)
    if(mobilenet_version == "Large"):
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            input_shape=input_shape,
            weights=pretrained_weights,
            classes=1000,
            include_preprocessing=False
            )
        base_model.trainable = True
        make_all_trainable(base_model)
        x = base_model(inputs, training = True)
        
    x.trainable = True
    x = layers.Flatten()(x)
    x = layers.Dense(512, name="dense_all_1_e", activation="relu")(x)
    if(use_dropout):
        x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, name="dense_all_2_e", activation="relu")(x)
    if(use_dropout):
        x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, name="dense_all_3_e", activation="relu")(x)
    outputs = layers.Dense(1, name="dense_output", activation="linear")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=(resolution,resolution,3,))

if(all_trainable):
    make_all_trainable(model)

use_learning_rate = tf.keras.optimizers.schedules.CosineDecay(args.initial_learning_rate, decay_steps=1000, alpha=0.0)
optim_ = keras.optimizers.Adam(
    learning_rate=use_learning_rate
)

if(args.optimizer == "RMSprop"):
    optim_ = keras.optimizers.RMSprop(
        learning_rate=use_learning_rate
    )

model.compile(
    optimizer=optim_,
    loss=mean_squere_percentage_error,
    metrics= metrics,
)
model.summary()
history = model.fit(
    train_batches,
    epochs=int(epochs / 2),
    validation_data=val_batches,
)

run.log("train_mse_50", history.history['mse'][-1])
run.log("train_mae_50", history.history['mae'][-1])
run.log("train_mae_cm2_50", history.history['mae_cm2'][-1])
run.log("train_mean_absolute_percentage_error_cm_50", history.history['mean_absolute_percentage_error_cm'][-1])

run.log("val_mse_50", history.history['val_mse'][-1])
run.log("val_mae_50", history.history['val_mae'][-1])
run.log("val_mae_cm2_50", history.history['val_mae_cm2'][-1])
run.log("val_mean_absolute_percentage_error_cm_50", history.history['val_mean_absolute_percentage_error_cm'][-1])

test_evaluation = model.evaluate(test_batches)
run.log("test_mse_50", test_evaluation[1])
run.log("test_mae_50", test_evaluation[2])
run.log("test_mae_cm2_50", test_evaluation[3])
run.log("test_mean_absolute_percentage_error_cm_50", test_evaluation[4])

history = model.fit(
    train_batches,
    epochs=int(epochs / 2),
    validation_data=val_batches,
)

run.log("train_mse_100", history.history['mse'][-1])
run.log("train_mae_100", history.history['mae'][-1])
run.log("train_mae_cm2_100", history.history['mae_cm2'][-1])
run.log("train_mean_absolute_percentage_error_cm_100", history.history['mean_absolute_percentage_error_cm'][-1])

run.log("val_mse_100", history.history['val_mse'][-1])
run.log("val_mae_100", history.history['val_mae'][-1])
run.log("val_mae_cm2_100", history.history['val_mae_cm2'][-1])
run.log("val_mean_absolute_percentage_error_cm_100", history.history['val_mean_absolute_percentage_error_cm'][-1])

test_evaluation = model.evaluate(test_batches)
run.log("test_mse_100", test_evaluation[1])
run.log("test_mae_100", test_evaluation[2])
run.log("test_mae_cm2_100", test_evaluation[3])
run.log("test_mean_absolute_percentage_error_cm_100", test_evaluation[4])

run.complete()