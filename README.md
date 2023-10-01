# Random Forest Classification Hit Predicton Model

## Overview

In this project, we have developed a Random Forest classification model with the primary goal of predicting whether a song has the potential to become a chart-topping hit. The project encompasses several key steps:

  - Data Collection: We gathered a diverse dataset comprising 2000 songs, which includes a wide range of genres, artists, and release years. Each song is represented in a structured format within a Pandas DataFrame, incorporating essential audio features.

  - Data Preprocessing: To prepare the data for model training, we conducted thorough data preprocessing. This phase involved feature standardization and transformation, ensuring that the audio features are in a consistent and analytically meaningful format.

  - Model Training: We employed a Random Forest classification algorithm to train our predictive model. This ensemble learning technique is well-suited for handling complex, multi-dimensional data and has a proven track record in various classification tasks.

  - Popularity Threshold: Defining the threshold for song popularity is crucial for our classification task. We set the threshold at a popularity score of 70, drawing insights from Spotify's popularity metric, which quantifies a song's current listening frequency.

  - Prediction: The trained model allows us to predict whether a given song is likely to achieve widespread acclaim and commercial success, classifying it as either a potential hit or not.

Our project aims to shed light on the factors that contribute to a song's popularity in today's dynamic music landscape. By combining machine learning techniques with comprehensive audio feature analysis, we aspire to provide valuable insights into what makes a song resonate with audiences and potentially top the charts. Whether you're a music enthusiast, industry professional, or data scientist, our project offers a compelling exploration of the intersection between music and data-driven predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Data Importing](#data-importing)
- [Model Creation](#model-creation)
- [Model Optimization](#model-optimization)
- [Collaborators](#collaborators)

## Introduction

In collaboration with our team, we have developed a robust Random Forest classification model designed for predicting the likelihood of a song becoming a chart-topping hit. This project involved the collection of data from a diverse selection of varying top 2000 songs, each meticulously organized into a Pandas DataFrame along with their corresponding audio features.

We leveraged advanced data preprocessing techniques, including feature standardization, to prepare the audio attributes of each song for model training. To define the criteria for a 'popular' song, we established a threshold of 70 for song popularity scores, derived from Spotify's proprietary metric that quantifies a song's popularity based on its current listening frequency.

Our model's primary objective is to discern whether a given song is likely to achieve widespread acclaim and commercial success. By harnessing the power of machine learning and data-driven insights, we aim to provide valuable predictions and contribute to a deeper understanding of the factors that influence song popularity in today's music landscape.

## Dependencies

Ensure you have these languages and libraries installed on your enviroment before installation

- Python
- Jupyter Notebook
- os
- seaborn
- spotipy
- requests
- pandas
- tensorflow
- spotipy
- matplotlib
- imblearn
- sklearn

## Installation

Clone the repository:
   ```sh
   git clone https://github.com/jnh17331/machine-learning-music.git
   cd machine-learning-music
  ```

## Usage
Predict Song Popularity:

In the model_opt1.ipynb notebook, you can fit the song or data you want to predict for hit potential. Follow these steps:

- Open the model_opt1.ipynb notebook using a Jupyter Notebook environment or a similar tool.

- In the model_opt1.ipynb notebook, locate the section where you can input the features or attributes of the song you want to predict for hit potential. You need to provide information such as song duration, acousticness, danceability, and other relevant features.

- Execute the notebook cell that predicts the song's hit potential using the pre-trained Random Forest classification model. The output will indicate whether the song is predicted to be a hit or not.

  ```python
  
  # Replace the sample values with your song data
  song_data = {
      "duration(ms)": 219724,
      "acousticness": 0.1690,
      "danceability": 0.511,
      ...
  }
  
  # Use the pre-trained model to predict hit potential
  prediction = rf_model.predict([list(song_data.values())])
  
  # Display the prediction (1 for hit, 0 for not)
  print("Predicted Hit Potential:", prediction[0])
  ```
  Replace the sample values in the song_data dictionary with your song's actual data.

- The output will indicate whether the song is predicted to be a hit or not based on the model's classification.

Feel free to adapt the provided code and notebooks to your specific song prediction needs, and explore the model's predictions for various songs or datasets of interest.
  

## Data Importing

  - Spotify API Data Retrieval:
    - Utilized Spotify's API to extract the top 100 tracks for each year, ranging from 2015 to the current date of September 25, 2023. This data was sourced from a variety of playlists identified by their unique playlist IDs.

  - Track ID Extraction:
    - Extracted the track IDs of the top 100 songs from each playlist to facilitate further data retrieval.

  - API Calls for Song Data:
    - Conducted two separate API calls to gather comprehensive data for each song. These API calls collected vital information, including the song's artist, release date, duration, and various audio analysis features.

  - Data Parsing and Aggregation:

    - Parsed the data for all retrieved tracks and meticulously organized it into structured pandas DataFrames.

    - Aggregated all individual CSV files, resulting from the API calls for different playlists, into a single cohesive dataset. This consolidated dataset serves as the foundation for subsequent analysis and model development.

## Model Creation

The heart of this project lies in the development of a Random Forest classification model that predicts whether a song will become a hit. Below are the steps involved in creating and training the model:

Feature Selection:

  - To create a robust model, carefully select the relevant features that influence a song's popularity. These features can include acousticness, danceability, energy, and more, depending on your dataset and goals. Below are the features used in our model:

![Audio Features](https://github.com/jnh17331/machine-learning-music/blob/main/Resources/audio_features.PNG?raw=true)

Data Preprocessing:

  - Clean and preprocess your dataset to handle missing values, outliers, and categorical variables. Ensure that the data is in a suitable format for training the machine learning model. Below is an example of our full dataset:

![Audio Features](https://github.com/jnh17331/machine-learning-music/blob/main/Resources/dataframe.PNG?raw=true)

Splitting the Dataset:

  - Divide your dataset into two subsets: a training set and a testing set. The training set is used to train the model, while the testing set helps evaluate its performance.

Model Selection:

  - Choose an appropriate machine learning algorithm for the classification task. In this project, we have used a Random Forest classifier, known for its ability to handle complex datasets and produce accurate predictions.
    - We have a pre-built model used for testing
    - And we have a dynamic model that can fit your data best based on feature selection
      
Prediction:

  - Once the model is trained and evaluated, you can use it to predict the hit potential of new songs by providing their features as input.

```python
# Example code for making predictions (inside the notebook)
song_data = {
    "duration(ms)": 219724,
    "acousticness": 0.1690,
    "danceability": 0.511,
    # Add more features as needed...
}

# Use the pre-trained model to predict hit potential
prediction = rf_model.predict([list(song_data.values())])

# Display the prediction (1 for hit, 0 for not)
print("Predicted Hit Potential:", prediction[0])
```

## Model Optimization

Ongoing

## Collaborators

- Jesse H. (jnh17331) - Data Importing, Model Creation, Visualizations
- Crystal B. (CBURKHARDT47) - Presentation, Data Importing, Ideation
- Charles M. (cnm92211) - Data Importing, Ideation, Bug Fixs
- Azkya S. (azkyasaid7-EfSpxQ) - Ideation + Bug Fixs
