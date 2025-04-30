# Knowledge Tracing

## Objectives

Train a KT (Knowledge Tracing) model to predict (the probability of) whether the student will answer a question correctly, given their historical interactions. The model chosen for this project was the SAKT (**S**elf-**A**ttentive model for **K**nowledge **T**racing).

TODO: Why SAKT was chosen

## Methodology

The SAKT model

Embedding of question IDs and interaction IDs.

## Results

### EDA

The dataset used is a subset of the [XES3G5M dataset](https://github.com/ai4ed/XES3G5M). These subsets were downloaded from:
* [XES3G5M_interaction_sequences](https://huggingface.co/datasets/Atomi/XES3G5M_interaction_sequences)
* [XES3G5M_content_metadata](https://huggingface.co/datasets/Atomi/XES3G5M_content_metadata)

Below are some high-level information about the subset:

* Number of interactions in the training set: 33397/18066 (0.006%)
* Number of unique questions: 7653/7652 (100%) (note: the extra index is `-1`, which is used for masking)
* Number of unique concepts: 866/865 (100%) (note: the extra index is `-1`, which is used for masking)
* Number of unique uid: 18066/18066 (100%)
* Dimension of the question content and concept content embeddings: 768

The training data is split into 5 folds with another set as the test set. Each fold containing approximately

* training set: folds 1-4, 26712 rows
* validation set: fold 5, 6685 rows
* test set: 3613 rows

None of the users (`uid`) span across splits, which would allow the model to better generalise to new users.

Data discrepancies
* There are

### Training

Early stopping with `patience` of 10. Validation accuracy is 

GTX1080 Ti

### Evaluation

Best model on the validation dataset is

This model is then trained on all 5 folds.

### Impact Analysis

## Concluding Remarks

### Strengths and Weaknesses

### Opportunities

* Richer data
* More time

Add lag time (time between question `timestamp`s) and elapsed time (the time taken to complete a question). [SAINT+](https://arxiv.org/abs/2010.12042) has shown that add these time-based features improved the model.

## Reproducibility

### Training the model locally

`./scripts/train_sakt_local.sh`