# 📊 Benchmarking Machine Learning and Transformer Models for Amazon Review Sentiment Classification 

This project explores a full end-to-end pipeline for sentiment analysis on text data using both traditional machine learning techniques and transformer-based large language models (LLMs). It includes text preprocessing, feature engineering, model training, evaluation, and robustness testing by adding noise to the data.

---

## 🧠 Table of Contents

1. Text Processing
2. N-Gram Function
3. WordCloud
4. Feature Engineering
5. TF-IDF Vectorization
6. Model
7. Accuracy
8. Hyperparameter Tuning
9. Best Model Based on Accuracy
10. Training & Testing
11. Logging with MLflow
12. DistilBERT LLM
13. DeBERTa-v3-base LLM
14. LLM Comparison
15. Add Noise and Re-Evaluate


---

## 📝 Text Processing
- Tokenization
- Stop-word removal
- Lemmatization
- Lowercasing

## 🔤 N-Gram Function
- Visualize n-grams (bigrams/trigrams) frequency
- Understand frequent word combinations

## ☁️ WordCloud
- Visual representation of word frequency

## 🛠 Feature Engineering
- Label creation (Positive vs Negative)
- Review length, polarity scores, etc.

## 🧮 TF-IDF Vectorization
- Transform text to numerical features for traditional ML models

## 🤖 Model
- Logistic Regression
- Random Forest
- SVM
- Naïve Bayes

## 📈 Accuracy
- Evaluate and compare model accuracies using accuracy score, F1-score

## 🎯 Hyperparameter Tuning
- Use `GridSearchCV` or other techniques for tuning model parameters

## ✅ Best Model Based on Accuracy
- Pick best parameters from tuning and retrain model

## 🧪 Training & Testing
- Train-test split
- Fit final model and evaluate

---

## 🔗 MLflow Tracking

All models and metrics are tracked using **MLflow**, including:

- Parameters (`max_depth`, `n_estimators`, etc.)
- Evaluation metrics (accuracy, loss, F1-score)
- Saved models and tokenizer
- Artifacts like confusion matrix plots and vectorizers

**MLflow Experiment Name**: `Sentiment-Analysis-Hyperparameter`  
**Repository Hosted on**: [DAGsHub](https://dagshub.com/siddharthadhikari85/sentimental_analysis)  
**Tracking URI**: `file:///kaggle/working/mlruns` (or local path during offline mode)

> You can view all runs, compare performance, and download artifacts via DAGsHub's MLflow dashboard.

---

## 🤗 Hugging Face Models

We used Hugging Face Transformers for fine-tuning and evaluation.

- ✅ **DistilBERT**: `distilbert-base-uncased`
- ✅ **DeBERTa-v3-base**: `microsoft/deberta-v3-base`

**Model Uploads**: Final models can be optionally pushed to Hugging Face Hub using:
```python
trainer.push_to_hub("Siddharth-Adhikari-07/finetuned-distilbert-sentiment")
trainer.push_to_hub("Siddharth-Adhikari-07/finetuned-deberta-sentiment")

```

> Note: Tokenizer and model artifacts are saved locally and logged in MLflow for reproducibility.

---

## 📋 Logging Train & Test Models

Every model's run includes:
- `.log_param` for all hyperparameters
- `.log_metric` for evaluation results
- `.log_artifacts` for saving:
  - Confusion Matrix
  - Model files
  - Tokenizer files
  - Preprocessing pipeline (Pickle)

---

## ✨ Results

| Model            | Accuracy |
|------------------|----------|
| LogisticRegression | 87.3%   |
| RandomForest     | 89.1%    |
| DistilBERT       | 91.8%    |
| DeBERTa-v3-base  | 93.5%    |

---

## 📦 Tech Stack

- Python 🐍
- Scikit-Learn
- Hugging Face Transformers
- MLflow + DAGsHub
- Matplotlib / Seaborn / Plotly
- Pandas, Numpy
- Streamlit (for optional deployment)
- Kaggle (for dataset and experimentation)

---

## 📌 Author

**Siddharth Adhikari**  
**Sathwick Kiran M S**
**Sahil Vachhani**

---
