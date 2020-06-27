# FakeNewsDetection
Using Machine Learning and NLP to segregate the unreliable and reliable news articles

## About this Project
This project analyzes about 18k news articles and their titles, lemmatizes and cleans them and then uses them as features, which are then used in Logistic Regression, to classify them as reliable (0) or 
not reliable (1). It is able to achieve a test set accuracy of 95.8%.

## Dataset
The dataset used is available on [Kaggle](https://www.kaggle.com/). It contains over 20000 news articles along with their authors, titles and labels classifying them
as reliable or not reliable.

It is present as train.csv in the repository. To access the data online, follow this [link](https://www.kaggle.com/c/fake-news/data).

## How to Run the Model on your System
1. Use this [link](https://www.kaggle.com/c/fake-news/data) to download the dataset and set the folder containing the downloaded data as the working directory. 

2. Make sure you have all the libraries used in the fake_news.py file. In case you need to download any of the libraries, use this command:
```
pip install 'your library name'
```

3. Once you have all the libraries imported, copy the code from fake_news.py and run it.


