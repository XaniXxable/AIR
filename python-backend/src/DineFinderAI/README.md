# Restaurant Recommendation System

This project leverages machine learning models and natural language processing techniques to provide restaurant recommendations, fine-tune ranking models and analyze tokens models. Below is an overview of the files and their functionalities.

## DatabaseManager.py
* **Purpose**: Stores restaurant relevant data from the yelp business dataset into a database.
* **Key Features**:
  * Reads json file 
  * Filters irrelevant entries (e.g. non-food businesses)

* **Usage**: 'poetry run ingest' -> yelp business dataset (described in the python-backend README.md) required in the resources folder

## SentimentModelTrainer.py
* **Purpose**: Trains a sentiment classification model to analyze customer reviews.
* **Key Features**:
  * Uses BERT for sentiment classification with three classes: Positive, Neutral, and Negative.
  * Supports k-fold cross-validation.
  * Generates classification reports for model evaluation.

* **Additional Notes**:
  * This model is trained via the restaurant_reviews_sample.csv file. This file was generated via ChatGPT AI to examine if a AI can generate valuable testdata for our AI model.
    * Outcome: No, the data is not suitable for training because the model is not able to classify the reviews correctly
  * Therefore, the model is not fully ready to predict the class of reviews.
  * Furthermore, this model needs the 'yelp_academic_dataset_review.json' file (also from the yelp business dataset zip file) in the resources folder.

## TokenAnalyser.py
* **Purpose**: Performs Named Entity Recognition (NER) on restaurant-related tokens.
* **Key Features**:
  * Uses BERT for token classification with labels like LOC, RES, CUISINE, etc.
  * Supports adding custom tokens for fine-tuning.
  * Implements k-fold cross-validation for model evaluation.