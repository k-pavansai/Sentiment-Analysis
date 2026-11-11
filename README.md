# Sentiment Analysis Using Machine Learning

## Dataset
`data/sentiment_data.csv` (241,145 rows and 3 columns in the supplied file; the unnamed index column is removed during cleaning).

## Target classes
The supplied Python experiment assumes:
- `0` = Negative
- `1` = Neutral
- `2` = Positive

Verify that this encoding matches the dataset source before interpreting results.

## Models
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

TF-IDF converts review text into numerical features. Models are compared using accuracy, macro precision, macro recall, macro F1-score and training time. The best model is selected by macro F1-score.

## Run
```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

The script saves model comparison and confusion-matrix plots in this folder.
