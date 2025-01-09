# Information Retrieval Course Project

## Project Overview

This repository contains the code, datasets, and instructions for the Advanced Information Retrieval course at TUGraz. The goal of this project is to implement and evaluate a **Advanced Information Retrieval System** capable of efficiently retrieving user based information.

---

## Table of Contents
1. [What We Did](#what-we-did)
2. [Datasets](#datasets)
3. [How to Train the Model](#how-to-train-the-model)
4. [Validation and Evaluation](#validation-and-evaluation)
5. [Requirements](#requirements)
6. [Usage](#usage)
7. [References](#references)

---

## What We Did

During this project, we:
- Implemented a **Transformer-based model BERT** to filter relevant information for the user input.
- Preprocessed text data with techniques like tokenization.
- Designed a pipeline to index, retrieve, and rank documents based on query relevance.

---

## Datasets

We used the following datasets for training and validation:
- **Dataset Name:** [Insert Dataset Name Here]
- **Source:** [Link to Dataset Source or Citation]
- **Description:** [Brief Description of the Dataset]

### Dataset Preprocessing
- Removed stop words, punctuation, and special characters.
- Performed stemming/lemmatization to normalize terms.
- Split data into training, validation, and testing sets.

---

## How to Train the Model

Follow these steps to train the retrieval model:

1. Clone the repository:
    ```bash
    git clone https://github.com/XaniXxable/AIR.git
    cd AIR
    ```

2. Install required dependencies:
    ```bash
    poetry install
    ```

3. Train the model:
    ```bash
    poetry run training
    ```

5. Checkpoints and logs will be saved in the `python-backend/resources/model` directory.

---

## Validation and Evaluation

The model was validated using **k-fold cross-validation**, which ensures robust evaluation by splitting the dataset into `k` subsets. Each subset served as a validation set once while the remaining `k-1` subsets were used for training. This process was repeated `k` times, and the final metrics were averaged across all folds. The best model is saved into `python-backend/resources/model/NER`.


### Validation Process:
1. After each training iteration:
   - Checked the queries that were not correctly parsed or retrieved.
   - Analyzed failure cases to identify patterns in retrieval errors.
   - Refined preprocessing steps and model parameters based on these insights.


## Requirements

To run this project, ensure the following requirements are met:

### System Requirements:
- **Python**: Version 3.10
- **Poetry**: Version 1.8.5

## Usage

Have a look at the [README](src/FastDineAPI/README.md).

## References

Add here