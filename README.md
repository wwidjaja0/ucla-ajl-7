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

# Results & Key Findings

# Impact Narrative
We believe that AI is a powerful tool we can use to help improve our healthcare system. We are confident that this project has the capability of reducing errors and biases integrated into our current AI models and can assist our healthcare workers in making quicker and more accurate diagnoses and treatment plans for underserved communities.

# Next Steps & Future Improvements
We hope to be able to revise our ResNet model by fine-tuning it such as by freezing more layers and adjusting the hyperparameters to see which provide the most optimal conditions for our dataset. We aim to improve its accuracy and ensure an overall better performing model across a diverse dataset of skin tones. We believe that by expanding our dataset to include an even larger area of skin conditions and types, we will be able to better encapsulate key characteristics of a skin condition across all skin tones. 

