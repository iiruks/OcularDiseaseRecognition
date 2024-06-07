Ocular Disease Recognition using VGG-19
This project aims to recognize ocular diseases, specifically cataracts, from fundus images using various Convolutional Neural Network (CNN) models. The models used include VGG-19, ResNet101, DenseNet101, VGG-16, ResNet50, and DenseNet121. This README provides an overview of the dataset preparation, model training, and evaluation processes.

# Table of Contents
Dataset Preparation
Model Training
Evaluation and Results
Installation and Usage

## Dataset Preparation

The dataset consists of fundus images labeled with diagnostic keywords for left and right eyes. The preprocessing steps involve loading and resizing the images.
```
cataract = np.concatenate((left_cataract, right_cataract), axis=0)
normal = np.concatenate((left_normal, right_normal), axis=0)

dataset = create_dataset(cataract, 1)
dataset = create_dataset(normal, 0)
```

## Model Training

The models used for training are VGG-19, ResNet101, DenseNet101, VGG-16, ResNet50, and DenseNet121. Each model is used for feature extraction, and a dense layer is added for classification. The models are compiled and trained with early stopping and model checkpoint callbacks.
```
vgg19 = create_model(VGG19(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))
resnet101 = create_model(ResNet101(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))
densenet101 = create_model(DenseNet101(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))
vgg16 = create_model(VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))
resnet50 = create_model(ResNet50(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))
densenet121 = create_model(DenseNet121(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)))

history = vgg19.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), verbose=1)
```

## Evaluation and Results
The models' performance is evaluated using metrics such as accuracy, loss, ROC curve, and AUC.

## Installation and Usage
To run this project, follow these steps:
Mount Google Drive:
```
from google.colab import drive
drive.mount('/content/gdrive')
```

Change the directory to the desired location:
```
import os
os.chdir('/content/gdrive/MyDrive')
```
Load the CSV file:
```
df = pd.read_csv("/content/gdrive/MyDrive/Ocular_Disease_Recognition/full_df.csv")
```

Preprocess the images and create the dataset:
```
dataset = create_dataset(image_category, label)
```

Train and evaluate the model:
```
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), verbose=1)
```
