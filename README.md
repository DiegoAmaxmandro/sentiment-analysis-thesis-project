# Customer Support Chat Sentiment Analysis

This project explores AI-related methods of customer dissatisfaction detection during customer support chats. It is the applied part of a research thesis to the MSc in Artificial Intelligence to the Irish National College.

## Project Objective

The objective is to categorize messages of customers as either satisfied or dissatisfied with a number of conventional machine learning and deep learning approaches. The paper examines various types of data representation, like raw text, BERT embeddings, PCA-reduced vectors as well as performance of models in different approaches.

## Project Structure

```plaintext
notebooks/
├── 01_preprocessing.ipynb        # Clean and prepare raw customer support data
├── 02_labelling.ipynb            # Generate weak supervision sentiment labels
├── 03_embeddings_pca.ipynb       # Generate sentence embeddings + apply PCA
├── 04_bilstm_model.ipynb         # Train BiLSTM model on tokenized text
├── 05_ml_models.ipynb            # ML models using PCA-reduced embeddings
├── 06_ml_raw_text_models.ipynb   # ML models using raw TF-IDF features
├── 07_generate_results_table.ipynb  # Summarize performance of all models

data/
├── raw/                          # Original dataset (e.g., twcs.csv)
├── processed/                    # Preprocessed, embedded, and labelled data
├── vader/                        # VADER sentiment scores
├── models/                       # Trained models (.h5)

results/
├── svm/, logreg/, rf/, bilstm/   # Model-specific predictions and outputs

outputs/
├── summary_results_metrics.csv   # Final metrics table
## Models Explored

- **SVM**, **Logistic Regression**, **Random Forest**
- **BiLSTM Neural Network**
- All models evaluated with 5-fold cross-validation.

---

## Tools & Libraries

- Python 3.9+
- scikit-learn
- pandas, numpy
- TensorFlow / Keras
- NLTK
- Sentence Transformers (BERT)
- Matplotlib, Seaborn

> Please see requirements.txt for a full list of dependencies.

## How to Reproduce

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
