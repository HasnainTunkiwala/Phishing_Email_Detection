# Phishing Email Detection

This project implements a machine learning-based system to detect phishing emails using TF-IDF feature extraction and classification models like Logistic Regression and Multinomial Naive Bayes.

---

## Features

- Preprocesses email data by combining subject and body text.
- Converts text data into TF-IDF features.
- Trains and evaluates two models: Logistic Regression and Naive Bayes.
- Compares models using Accuracy, Precision, Recall, and F1-score.
- Allows classification of new emails interactively.
- Fully runnable on Google Colab with minimal setup.

---

## Dataset

The project uses the [CEAS_08.csv](path-to-dataset-if-online) dataset, which contains labeled emails as phishing (1) or legitimate (0).

---

## Getting Started

1. Clone this repository or open it directly in [Google Colab](https://colab.research.google.com/github/HasnainTunkiwala/Phishing_Email_Detection).

2. Upload the dataset (`CEAS_08.csv`) to the Colab environment or mount your Google Drive.

3. Run the notebook cells to preprocess data, train models, and evaluate performance.

4. Use the provided example to classify new emails.

---

## Requirements

- Python 3.x  
- pandas  
- scikit-learn  

You can install the required libraries with:

```bash
pip install pandas scikit-learn
