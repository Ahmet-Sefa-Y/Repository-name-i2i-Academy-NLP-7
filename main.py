import re
from pathlib import Path

import pandas as pd
from textblob import TextBlob


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "i", "my", "we",
    "our", "you", "your", "they", "their", "this", "these", "those"
}


def clean_text(text):
    """Converts text to lowercase, removes punctuation, and filters stop words."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)


def get_sentiment_label(score):
    """Labels polarity scores as Positive, Negative, or Neutral."""
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


def main():
    dataset_path = "yelp_reviews.csv"
    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)

    reviews = pd.read_csv(dataset_path)

    print(f"Total reviews loaded: {len(reviews)}")
    print("\nDataset columns:")
    print(reviews.columns.tolist())

    reviews["clean_text"] = reviews["text"].apply(clean_text)
    reviews["sentiment_score"] = reviews["clean_text"].apply(
        lambda review: TextBlob(review).sentiment.polarity
    )
    reviews["sentiment_label"] = reviews["sentiment_score"].apply(
        get_sentiment_label
    )

    print("\nFirst 5 processed reviews:")
    preview = reviews[
        ["text", "clean_text", "sentiment_score", "sentiment_label"]
    ].head().copy()

    preview["text"] = preview["text"].str[:70] + "..."
    preview["clean_text"] = preview["clean_text"].str[:70] + "..."

    print(preview.to_string(index=False))

    print("\nSentiment statistics:")
    sentiment_counts = reviews["sentiment_label"].value_counts()
    print(sentiment_counts)

    print("\nSentiment percentages:")
    sentiment_percentages = (
        reviews["sentiment_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
    print(sentiment_percentages)

    output_path = output_directory / "sentiment_results.csv"
    reviews.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()