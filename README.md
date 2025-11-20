markdown
# Fake News Detection - Natural Language Processing Project

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a **Fake News Detection System** using Natural Language Processing (NLP) and machine learning techniques. The system classifies news headlines as either "Fake" or "Real" using various ML models with TF-IDF feature extraction.

**Key Features:**
- Text preprocessing and feature engineering
- Multiple ML model comparison
- Hyperparameter tuning for optimal performance
- Comprehensive model evaluation
- Production-ready pipeline for predictions

## 📊 Dataset Description

The dataset contains news headlines from Reddit's world news channel with binary classification labels:

- **Source**: Reddit world news channel
- **Features**:
  - `headline`: Text content of news headlines
  - `fakeornot`: Binary label (0 = Fake News, 1 = Real News)
- **Size**: Tab-separated training data file
- **Class Distribution**: Balanced dataset for reliable model training

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install pandas matplotlib scikit-learn nltk xgboost jupyter
```

### NLTK Downloads
Run the following in Python to download required NLTK resources:
```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords') 
nltk.download('omw-1.4')
```

## 🛠️ Usage

1. **Clone the repository** and navigate to the project directory
2. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook
   ```
3. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Preprocess text data
   - Train multiple ML models
   - Evaluate model performance
   - Make predictions on test data

## 🔬 Methodology

### Data Preprocessing
- Text cleaning (special characters, digits removal)
- Lemmatization using WordNet
- Stopword removal
- TF-IDF vectorization with n-grams (1,2)

### Models Implemented
- Logistic Regression
- Naive Bayes (Multinomial)
- Random Forest
- Linear SVC
- XGBoost

### Model Selection
Comprehensive comparison of multiple algorithms with hyperparameter tuning using GridSearchCV.

## 📈 Results

### Best Performing Models
1. **Logistic Regression (Tuned)**: Highest accuracy with optimized hyperparameters
2. **LinearSVC (Tuned)**: Strong performance with fast training time
3. **Baseline Logistic Regression**: Reliable performance with minimal configuration

### Key Findings
- Text preprocessing (lemmatization, stopword removal) improved model performance
- TF-IDF with n-grams (1,2) captured important contextual features
- Logistic Regression consistently performed well across different configurations
- Hyperparameter tuning provided significant improvements in accuracy

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Tuned) | ~94% | - | - | - |
| LinearSVC (Tuned) | ~93% | - | - | - |
| Baseline Logistic Regression | ~92% | - | - | - |

*Note: Exact metrics may vary based on random state and data splits*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional model implementations
- Feature engineering improvements
- Performance optimizations
- Documentation enhancements

## 📄 License

This project is for educational purposes as part of the IronHacks Week 7 NLP Project.

---

**Note**: Replace file paths in the code with your actual dataset locations before running.
