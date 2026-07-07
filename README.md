# NLP Sentiment Analysis

This project was developed for the i2i internship program to analyze customer review sentiment using Python and Natural Language Processing.

## Technologies

- Python
- Pandas
- TextBlob

## Dataset

The project uses a Yelp review dataset containing 10,000 English customer reviews.

## Project Features

- Loads review data from a CSV file using Pandas
- Cleans review text by converting it to lowercase, removing punctuation, and filtering common stop words
- Calculates sentiment polarity scores using TextBlob
- Labels reviews as Positive, Neutral, or Negative
- Saves processed results to `outputs/sentiment_results.csv`

## Sentiment Labels

- Positive: score greater than 0.1
- Negative: score lower than -0.1
- Neutral: score between -0.1 and 0.1

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

## Project Structure

```text
i2i-Academy-NLP-7
├── main.py
├── yelp_reviews.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Output

The application creates `outputs/sentiment_results.csv`, which includes the original review text, cleaned text, sentiment score, and sentiment label.
