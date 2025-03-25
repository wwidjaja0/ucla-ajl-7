# Ucla-AJI-7
GitHub repository for the Break Through Tech AI @ UCLA Kaggle competition Spring 2025.

# Team Members
We are a team of 6 from UCLA and UCSD studying Computer Science and Mathematics.

Our team consisted of [William Widjaja](https://github.com/wwidjaja0) who contributed significantly on the data preprocessing stage in collaboration with [Rita Sargsyan](https://github.com/RitaSargsyan). Widjaja further made major contributions during the model training and evaluation phase. [Phi Hung Nguyen](https://github.com/AzineZ ) assisted in creating a baseline model to help the team gauge how our data would perform. [Mahsa Ileslamlou](https://github.com/mahsailes) researched data preprocessing measures to help reduce bias within our model and determined the model we used which had been shown to perform well with dermatology in AI with support of [Parnian Kashani](https://github.com/ghapanda) and [Eric Lu](https://github.com/ericslu).

# Project Overview
There has been a pattern of dermatology AI tools underperforming for people of different skin colors as a result of the poor diversity of training data. This has played a detementrial effect on marginalized and underserved communities from delayed treatments and other health disparities.

This project is in collaboration with Break Through Tech, a Cornell Tech initiative that connects undergraduate students from diverse backgrounds to impactful tech opportunities and the Algorithmic Justice League, an organization that merges art and research to bring awareness on the social impacts and potential harms of artificial intelligence.

The objective of this competition is to develop a machine learning model that will be equipped to classify 21 skin additions across a diverse skin tone data set. 

By developing a model that is able to accurately identify skin conditions across all skin tones, this project can play a significiant role in reducing health disparities in dermatology, aim to provide better treatment for all individuals, and overall contribute to developing more accurate AI healthcare models.

# Project Highlights
Our project focused on developing a machine learning model with minimal bias using *ResNet152V2* to classify **21** different skin conditions across the diverse set of skin tones provided to us via Kaggle. We implemented techniques such as data augmentation in which we resized and adjusted the brightness and contrast of the images and added custom layers on top of the base model to improve the model's performance. Our major findings indicate that our model was able to begin learning, however, there are necessary adjustments needed to be made to help improve its accuracy. We identified areas of improvement such as implementing more data augmentation techniques and fine-tuning the model more. 

# Setup & Execution
## ✅ **Setup & Execution Instructions**
### 📥 **1. Environment Setup**
- Recommended: Use **Google Colab** for easy environment management.
- If running locally, ensure the following packages are installed:
```bash
pip install kaggle pandas numpy opencv-python matplotlib seaborn albumentations tensorflow scikit-learn
```

### 📂 **2. Data Download**
- Place your `kaggle.json` API key in your working directory.
- Run the following (already in the notebook) to download and unzip the dataset:
```python
! pip install -q kaggle
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c bttai-ajl-2025
! unzip -q bttai-ajl-2025.zip
```
This downloads `train.csv`, `test.csv`, and images into your workspace.

### 🔎 **3. Data Preprocessing & Augmentation**
#### 🔎 **Load `train.csv` and `test.csv` into Pandas DataFrames**
```python
import pandas as pd

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Add .jpg extension to reference the image files
train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'
test_df['md5hash'] = test_df['md5hash'].astype(str) + '.jpg'
```

#### 🔗 **Create the `file_path` Column for Image Access**
This combines the `label` and `md5hash` to create a relative image path:
```python
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']
```

#### 🖼 **Apply Albumentations-Based Data Augmentation**
To increase training data diversity, apply random transformations and save the augmented images:
```python
import albumentations as A
import cv2
import os
import numpy as np

transform = A.Compose([
    A.Resize(128, 128),
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

def augment_and_save(image_path, output_path, transform, num_augmentations=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(num_augmentations):
        augmented = transform(image=image)
        augmented_image = (augmented['image'] * 255).astype(np.uint8)
        cv2.imwrite(f"{output_path}_aug_{i}.png", cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
```

#### 📂 **Run Augmentation Loop**
Iterate through all images and generate augmented samples:
```python
base_dir = './train/train'
output_base_dir = './train/augmented_train'

for skin_type in os.listdir(base_dir):
    os.makedirs(os.path.join(output_base_dir, skin_type), exist_ok=True)
    for image_name in os.listdir(os.path.join(base_dir, skin_type)):
        image_path = os.path.join(base_dir, skin_type, image_name)
        output_image_path = os.path.join(output_base_dir, skin_type, image_name.split('.')[0])
        augment_and_save(image_path, output_image_path, transform)
```

#### 🎲 **Randomly Select One Augmented Image per Sample**
For training, randomly pick **one** augmented version for each original image:
```python
import random

augmented_base_dir = './train/augmented_train/'
random_augmented_paths = []

for _, row in train_df.iterrows():
    label = row['label']
    base_filename = os.path.splitext(os.path.basename(row['file_path']))[0]
    aug_dir = os.path.join(augmented_base_dir, label)
    possible_files = [f for f in os.listdir(aug_dir) if f.startswith(base_filename + '_aug_')]
    
    if possible_files:
        chosen_file = random.choice(possible_files)
        random_augmented_paths.append(os.path.join(label, chosen_file))
    else:
        random_augmented_paths.append(row['file_path'])  # Fallback to original

train_df['random_augmented_file_path'] = random_augmented_paths
```

#### 🧼 **Normalize Paths (Optional but Recommended)**
Ensure paths are relative for Keras’ `flow_from_dataframe()`:
```python
base_dir = './train/augmented_train/'
train_df['random_augmented_file_path'] = train_df['random_augmented_file_path'].apply(
    lambda x: x.replace(base_dir, '') if x.startswith(base_dir) else x
)
```

# Data Exploration
## Dataset:
- The dataset: it is a smaller portion of the FitzPatrick17k collection, which contains roughly 17,000 labeled images showcasing both serious and cosmetic skin conditions across a spectrum of skin tones, as categorized by the FitzPatrick skin tone scale (FST). This subset includes approximately 4,500 images covering 21 skin conditions out of the 100+ found in the full dataset.
- No other dataset was used beside the one given on Kaggle.

## Data Exploration and Approaches: 
### 🔢 Label Encoding
- Categorical labels are converted into integer format using `LabelEncoder`.
- This transformation ensures that labels are machine-readable and compatible with model training.

### ✂️ Train/Validation Split
- The dataset is split into training and validation sets using `train_test_split`.
- A standard 80/20 split is applied.
- A fixed `random_state` ensures reproducibility of results.

### 🖼️ Image Normalization
- Images are normalized using `ImageDataGenerator(rescale=1./255)`.
- This rescales pixel values to the range `[0, 1]`.
- Normalization improves model convergence and stability during training.

### 🔄 Data Generation Setup
- A custom function `create_generator()` is used to create image generators.
- It leverages `flow_from_dataframe` to:
  - Load images dynamically from directories.
  - Match them with their corresponding labels using dataframe columns.

### 🧪 Test Data Preparation
- Test data is handled separately with the `preprocess_test_data()` function.
- This function:
  - Uses `ImageDataGenerator` for normalization.
  - Sets `class_mode=None` since test labels are not available.

# Visualizations from Exploratory Data Analysis: 
<img src="https://drive.google.com/uc?export=view&id=1hbWVETu4vT4PCbxlE0Cb0gWqpIQEKZ98" alt="Image" width="800"/>
<img src="https://drive.google.com/uc?export=view&id=19tzBmDF0rSRJOKOpB04xNym4-RJkDe2v" alt="Image" width="800"/>

# Model Development
## 🧠 **Model Training**
### 🛠 **Prepare Keras Data Generators**
Finally, create Keras `ImageDataGenerator` objects:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='./train/augmented_train/',
    x_col='random_augmented_file_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    validate_filenames=False
)
```

- The model uses **ResNet152V2** pretrained on ImageNet with custom dense layers.
- To start training:
```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)
```
- After training, the model is saved:
```python
model.save("resnet152v2.h5")
```

### 📈 **Model Evaluation & Prediction**
- Load the model and make predictions on the test set:
```python
model = load_model("./resnet152v2.h5")
test_generator = preprocess_test_data(test_df, test_dir)
predictions = model.predict(test_generator)
```
- Convert predictions to labels and save:
```python
submission.to_csv("submission.csv", index=False)
```

### 🗂 **Output**
- Final predictions are saved in **`submission.csv`** with class labels ready for competition submission.

# Results & Key Findings
## 📊 **1. Dataset Imbalance Observed**
- The dataset shows **class imbalance** across skin condition labels, particularly when broken down by the **Fitzpatrick skin scale**.
- Certain conditions (e.g., *prurigo-nodularis*) are **overrepresented**, while others have **limited samples**, making balancing important.
- Fitzpatrick types I-III dominate the dataset, suggesting fewer examples for very dark (IV-VI) skin tones.

## 🔄 **2. Need for Data Augmentation**
- Augmentation is crucial to synthetically expand minority classes and prevent overfitting.
- Applying **random rotations, brightness/contrast adjustments, and horizontal flips** can increase data variety and model robustness.
- This step can help balance rare classes and diverse skin tones in the training data.

## 🧠 **3. Model Performance vs. Complexity**
- Initial model choice, **ResNet152V2**, provided good feature extraction power but was heavy and slow to train.
- Future iterations could explore **MobileNetV2** or **EfficientNetB0** for similar accuracy with faster training time and less resource usage.

## 📈 **4. Successful End-to-End Pipeline Execution**
- The team successfully:
  - Loaded and preprocessed the data
  - Augmented training images dynamically
  - Set up a Keras data generator pipeline
  - Trained a ResNet152V2-based CNN model
  - Generated predictions and prepared a valid submission file

## 🔍 **5. Recommendations for Improvement**
- Consider **stratified sampling** when splitting train/validation sets to preserve class distribution.
- Investigate **fine-tuning** the model by unfreezing top ResNet layers after initial training for better accuracy.
- Incorporate **Fitzpatrick scale-based stratification** or balancing strategies for fairer performance across skin tones.

# Impact Narrative
We believe that AI is a powerful tool we can use to help improve our healthcare system. We are confident that this project has the capability of reducing errors and biases integrated into our current AI models and can assist our healthcare workers in making quicker and more accurate diagnoses and treatment plans for underserved communities.

# Next Steps & Future Improvements
We hope to be able to revise our ResNet model by fine-tuning it such as by freezing more layers and adjusting the hyperparameters to see which provide the most optimal conditions for our dataset. We aim to improve its accuracy and ensure an overall better performing model across a diverse dataset of skin tones. We believe that by expanding our dataset to include an even larger area of skin conditions and types, we will be able to better encapsulate key characteristics of a skin condition across all skin tones. 

