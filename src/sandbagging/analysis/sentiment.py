from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

model_path = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


def get_sentiment(text):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = np.exp(scores) / np.exp(scores).sum()  # softmax

    weighted_score = (-1 * scores[0]) + (0 * scores[1]) + (1 * scores[2])

    print(f"Text: {text}, Sentiment: {weighted_score}")
    return float(weighted_score)
