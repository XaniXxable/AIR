# Restaurant Recommendation System

This project leverages machine learning models and natural language processing techniques to provide restaurant recommendations, fine-tune ranking models and analyze tokens models. Below is an overview of the files and their functionalities.

## RankingModel.py
* **Purpose**: Re-ranks candidate results based on additional factors like reviews, scores, and comments.
* **Key Features**:
  * Extends the DineFinder class to include re-ranking logic.
  * Utilizes external factors (e.g., comments, ratings) to improve result ranking.

* **Usage**: Extend or use this class to implement custom ranking logic.

## SentimentModelTrainer.py
* **Purpose**: Trains a sentiment classification model to analyze customer reviews.
* **Key Features**:
  * Uses BERT for sentiment classification with three classes: Positive, Neutral, and Negative.
  * Supports k-fold cross-validation.
  * Generates classification reports for model evaluation.

## TokenAnalyser.py
* **Purpose**: Performs Named Entity Recognition (NER) on restaurant-related tokens.
* **Key Features**:
  * Uses BERT for token classification with labels like LOC, RES, CUISINE, etc.
  * Supports adding custom tokens for fine-tuning.
  * Implements k-fold cross-validation for model evaluation.