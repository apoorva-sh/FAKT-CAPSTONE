# UW-MSDS Capstone 2020

This github repository holds all the code and documentation for UW Data Science Final Capstone Project. The team members part 
of this project are Frank Chen, Apoorva Shetty, Kamala V J, Tharun Sikhinam.

# D-Classified News

## Introduction
This capstone project is an undertaking by the students of UW in their data science capstone class.

### Problem Statement
Noonum is a fin-tech AI startup that leverages graphs and NLP to be a knowledge engine for business and finance. Their current news dashboard (As seen below) contains both relevant and irrelevant news articles. The aim of this project is to classify news articles as relevant or irrelevant based on their "market-relevance" and to explain why this classification was made, so as to better service their clients with their news dashboard.

![Noonum Dashboard](docs/images/NoonumDashboard.png)

### Proposed Solution

Our proposed solution involved four major components plugged together to create a fully functional classification and explanation model.

![System Design](docs/images/SystemDesign.png)

Our basic concept involved processing the raw text to remove stop words and perform other cleaning tasks before feeding it into the feature engineering module, from where our training model would learn to classify each article. These classified articles would be fed into an explanation model to generate individual explanation for each articles. 

The insights that we gained using this explanation would be used to aggressively clean our data and to re-label our raw data to improve model performance.

This repo contains these four modules and can be executed to get the results we obtained

## Data

![Data Description](docs/images/Data.png)

## Folder Structure

```
.
├── README.md
├── data
│   ├── clean
│   │   ├── irrelevant_news.json
│   │   └── relevant_news.json
│   ├── feature
│   │   ├── test_lda.csv
│   │   ├── test_{feature}.csv
│   │   ├── train_lda.csv
│   │   └── train_{feature}.csv
│   └── raw
│       ├── irrelevant_articles.json
│       └── relevant_articles.json
├── docs
│   ├── Final_Project_Report.docx
│   ├── Poster_Final_FAKT.png
│   └── images
│       ├── Data.png
│       
├── models
│   ├── bow_logistic_model.sav
│   ├── bow_xgboost_model.sav
│   ├── lda_bow
│   └── lda_tfidf
└── src
    ├── explanation
    │   ├── bow_xgboost.ipynb
    │   └── doc2vec_xgboost_NB_LR.ipynb
    ├── feature_engineering
    │   ├── doc2vec_embedding.ipynb
    │   ├── LDA_feature_extraction.ipynb
    │   ├── bow.ipynb
    │   ├── ner-entity.ipynb
    │   └── tf-idf.ipynb
    ├── model
    │   └── Model_Training.ipynb
    └── preprocess
        ├── Capstone_Descriptive.ipynb
        └── Preprocessing.ipynb
```

## How to use this repository

The repository has four main folders. 
1. [data](./data) - holds all the raw, clean and feature sets used for model training
2. [src](./src) - holds all the python notebooks used in the project
3. [docs](./docs) - contains the project report and poster
4. [models](./models) - all trained models are saved in this folder

#### Descriptive Statistics 
The descriptive stats are performed in [Capstone_Descriptive.ipynb](./src/preprocess/Capstone_descriptive.ipynb) notebook.
#### Preprocess the data
Pre-processing is done under [src/preprocess](./src/preprocess)
#### Feature Engineering
Each Feature Engineering step is broken down into its own notebook. They are under [src/feature_engineering](./src/feature_engineering). BoW and Tf-IDF generate pickle files which are used to generate the features, it is not feasible to store the feature vector as .csv file.
#### Train or use pre-trained Models
Models are trained in [src/model/Model_Training.ipynb](./src/model/Model_Training.ipynb) and the trained models are saved under [models](./model) directory of the form {feature_type}_{model_name}.sav. 
#### Generating Explanations
Explanation and results of the analysis are documented under [src/explanations](./src/explanation)

The results of the best model and explanation can be found at [src/explanation/bow_xgboost.ipynb](./src/explanation/bow_xgboost.ipynb)

## Conclusion
- Simpler feature sets such as BoW and TF-IDF performed well. Aggressive cleaning & pre-processing w.r.t to the context of the application improved the accuracy of the model.
- LIME gave us a clearer picture of what our model was truly learning; indicating that the market relevant terms were being captured by the models. 
- The task lying ahead is to provide for a feedback loop to re-incorporate what we learn from our explanation into labelling the data, and also into our data pre-processing and feature-engineering steps.

