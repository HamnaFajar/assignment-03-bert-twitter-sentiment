# Assignment 03 – BERT Training on Twitter Sentiment Dataset + PyQt GUI

**Course:** Natural Language Processing Lab
**University:** Shifa Tameer-e-Millat University, Islamabad

## Dataset Source
Twitter Sentiment Analysis dataset (CSV format), e.g. one of:
- Kaggle: "Twitter US Airline Sentiment" – https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Kaggle: "Sentiment140" – https://www.kaggle.com/datasets/kazanova/sentiment140

Download the CSV, rename the text column to `text` and the label column to `sentiment`
(or edit `TEXT_COL` / `LABEL_COL` at the top of `train_bert.py` to match your file),
then place it at `dataset/twitter_sentiment.csv`.

## Project Structure
```
assignment_03_bert_twitter_sentiment/
|-- train_bert.py          # trains BERT, saves graphs + model
|-- app.py                 # PyQt5 GUI
|-- requirements.txt
|-- README.md
|-- dataset/
|   |-- twitter_sentiment.csv
|-- saved_bert_model/      # created automatically after training
|-- results/               # graphs created automatically after training
|-- screenshots/
```

## Environment Setup

1. **Training (needs a GPU — do this on Google Colab, it's free):**
   - Open https://colab.research.google.com, new notebook
   - Runtime > Change runtime type > GPU
   - Upload `train_bert.py` and your `dataset/twitter_sentiment.csv`
   - In a Colab cell run:
     ```
     !pip install transformers torch pandas scikit-learn matplotlib seaborn
     !python train_bert.py
     ```
   - This produces `saved_bert_model/` (config.json, pytorch_model.bin, tokenizer files,
     label_map.json) and `results/` (4 graphs). Download both folders to your PC.

2. **Running the GUI (this runs fine on a normal laptop, CPU only, no GPU needed):**
   - Create a virtual environment (recommended): `python -m venv venv`
   - Activate it, then:
     ```
     pip install -r requirements.txt
     python app.py
     ```
   - In the GUI:
     - Click **Load Dataset** → pick `dataset/twitter_sentiment.csv`
     - Click **Load Model** → pick the `saved_bert_model/` folder (downloaded from Colab)
     - Click any tweet row → prediction appears
     - Or type a sentence in the text box → click **Predict**

## Evaluation Metrics
Reported in `results/`: accuracy, precision, recall, F1-score (weighted), confusion matrix,
training/validation loss curve, training/validation accuracy curve, class distribution graph.

## Paul's Critical Thinking Standards
- **Clarity:** Dataset has two columns — `text` (tweet content) and `sentiment`
  (positive/negative/neutral or positive/negative).
- **Accuracy:** Metrics reported as produced by the evaluation script, including any
  weaknesses (e.g. lower recall on the minority class).
- **Precision:** Model = `bert-base-uncased`, 80/20 train-test split, 3 epochs, batch size 16.
- **Relevance:** Loss curve, accuracy curve, and confusion matrix graphs directly support
  the reported model performance.
- **Depth:** Misclassifications are usually short tweets, sarcasm, or slang that BERT's
  base tokenizer doesn't handle well without more fine-tuning data.
- **Logic:** Train → save → load in GUI → predict pipeline is fully connected end-to-end.
- **Fairness:** Note any class imbalance in the dataset (e.g. more positive tweets than
  neutral) and that slang/abbreviations common in tweets may reduce accuracy.

## Notes
- Do not use a pre-built sentiment pipeline — this project fine-tunes BERT on the
  chosen dataset from scratch, as required.
- GUI never hardcodes predictions — every prediction is a live call to the loaded model.
