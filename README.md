# Knowledge Tracing

## Objectives

Train a KT (Knowledge Tracing) model to predict (the probability of) whether the student will answer a question correctly, given their historical interactions. The model chosen for this project was the SAKT (**S**elf-**A**ttentive model for **K**nowledge **T**racing).

TODO: Why SAKT was chosen

## Methodology

The SAKT model with the following parameters:

TODO: Add list of parameters

Embedding of question IDs and interaction IDs.

An embedding dimension of 768 (which is different to those used in the paper) was chosen to match the pre-embedding dimension of the questions and content, i.e. to allow a direct sum of the two.  

TODO: explain deviations from the original paper

A slightly modified version of the SAKT model was also trained. The question pre-embeddings were added to the learned question embeddings. This adds the questions' content information into the learned embedding before the attention section. There are several strategies to incorporate existing embeddings, however, addition was selected due to ease of implementation and the fact that it doesn't require additional parameters, which contributes to overfitting.

The pre-embeddings are registered as a buffer in the model and hence are not updating during training.

## Results

The best validation result came from the SAKT model with question pre-embeddings; hyperparameters are listed below.

TODO: list hyperparameters

The same model was used 

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

* `concept_embeddings` contained 1175 row whereas there are only 866 unique `concepts`; the latter is consistent with the original paper of 865 (note the extra ID is `-1`, which is reserved for padding). Due to this discrepancy, the `concept_embeddings` was not used.

### Training

Early stopping with `patience` of 10. Validation accuracy is 

GTX1080 Ti

Multiple training sessions were done to understand the learning curve, i.e. the impact of 

### Evaluation

#### Dataset Size

For the purposes of this exercise, 
Best model on the validation dataset is

This model is then trained on all 5 folds.

### Impact Analysis

## Concluding Remarks

### Strengths and Weaknesses



### Opportunities

1. Train the model on all the 5 folds and recalculate the test metrics.
1. Increase the data
1. Concatenate the external embeddings instead of adding them.
1. Add lag time (time between question `timestamp`s) and elapsed time (the time taken to complete a question). [SAINT+](https://arxiv.org/abs/2010.12042) has shown that add these time-based features improved the model.
1. Use sinusoidal encoding for the positions, rather than a learned embedding. This would reduce the number of parameters in the model.

## Reproducibility

### Training the model locally

`./scripts/train_sakt_local.sh`