# Ucla-AJI-7
GitHub repository for the Break Through Tech AI @ UCLA Kaggle competition Spring 2025.

# Team Members
[William Widjaja](https://github.com/wwidjaja0), [Rita Sargsyan](https://github.com/RitaSargsyan), [Phi Hung Nguyen](https://github.com/AzineZ ), [Mahsa Ileslamlou](https://github.com/mahsailes), [Parnian Kashani](https://github.com/ghapanda), [Eric Lu](https://github.com/ericslu)

# Project Overview
There has been a pattern of dermatology AI tools underperforming for people of different skin colors as a result of the poor diversity of training data. This has played a detementrial effect on marginalized and underserved communities from delayed treatments and other health disparities.

This project is in collaboration with Break Through Tech, a Cornell Tech initiative that connects undergraduate students from diverse backgrounds to impactful tech opportunities and the Algorithmic Justice League, an organization that merges art and research to bring awareness on the social impacts and potential harms of artificial intelligence.

The objective of this competition is to develop a machine learning model that will be equipped to classify 21 skin additions across a diverse skin tone data set. 

By developing a model that is able to accurately identify skin conditions across all skin tones, this project can play a significiant rule in reducing health disparities in dermatology, aim to provide better treatment, and develop more accurate AI healthcare models.

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

