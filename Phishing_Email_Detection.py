import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load and Explore Data
df = pd.read_csv("/content/CEAS_08.csv")

# Display the first 5 rows
print("First 5 rows of the DataFrame:")
display(df.head())

# Get information about the DataFrame
print("\nDataFrame Info:")
df.info()

# Get descriptive statistics of the DataFrame
print("\nDescriptive Statistics:")
display(df.describe(include='all'))

# Check the distribution of the target variable
print("\nDistribution of the Target Variable:")
display(df['label'].value_counts())


# 2. Data Preprocessing and Feature Engineering
# Combine 'subject' and 'body' into a single 'text' column
df['subject'] = df['subject'].fillna('')
df['text'] = df['subject'] + ' ' + df['body']

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

print("Shape of the TF-IDF matrix:", X.shape)


# 3. Model Training and Selection
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
print("\nLogistic Regression model trained.")

# Train Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)
print("Multinomial Naive Bayes model trained.")


# 4. Model Evaluation
# Logistic Regression Evaluation
lr_predictions = logistic_regression_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1-score: {lr_f1}")

# Naive Bayes Evaluation
nb_predictions = naive_bayes_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

print("\nNaive Bayes Model Evaluation:")
print(f"Accuracy: {nb_accuracy}")
print(f"Precision: {nb_precision}")
print(f"Recall: {nb_recall}")
print(f"F1-score: {nb_f1}")

# 5. Finish Task
print("\nConclusion:")
if lr_f1 > nb_f1:
    print("Logistic Regression performs better with a higher F1-score.")
    best_model = logistic_regression_model
else:
    print("Naive Bayes performs better with a higher F1-score.")
    best_model = naive_bayes_model

# Example of classifying a new email
new_email = ["You are receiving this email because you joined the Kerzner International Talent Community on 7/28/25. You will receive these messages every 7 day(s). Your Job Alert matched the following jobs at jobs.kerzner.com."]
new_email_tfidf = tfidf_vectorizer.transform(new_email)
prediction = best_model.predict(new_email_tfidf)

print(f"\nNew email classification: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
