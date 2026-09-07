# 💊 Medicine Recommendation & Treatment Prediction System

A data science project that explores a 100,000-record medicine dataset and builds recommendation and prediction models to suggest treatments and medicines based on disease, treatment type, or tablet name.

## 📋 Overview

This project analyzes a large medical dataset (100,000 rows × 11 columns) containing information about diseases, treatments, tablets, dosages, side effects, pricing, and more. It combines exploratory data analysis (EDA) with multiple machine learning approaches to:

- Search and filter medicines by disease, treatment, or tablet name
- Recommend the most common Disease → Treatment → Tablet combinations
- Predict the most likely treatment for a given disease/tablet using content-based similarity and classification models

## 📊 Dataset

The dataset (`medicine_dataset.csv`) contains the following columns:

| Column | Description |
|---|---|
| Disease | Name of the medical condition |
| Treatment | Type of therapy (e.g., Antipyretic Therapy, Analgesic Therapy) |
| Tablet Name | Name of the medicine/drug |
| Dosage | Dosage amount |
| Time Taken | When the medicine should be taken |
| Effect Time Duration | How long the effect lasts |
| Which Age | Target age group |
| Company | Manufacturer |
| Purity | Purity percentage |
| Side Effects | Reported side effects |
| Price | Price in ₹ (INR) |

**Size:** 100,000 records covering 25 diseases and 24+ treatment types.

## 🔍 Exploratory Data Analysis

- Distribution of records across diseases, treatments, and tablets
- Frequency of Disease → Treatment → Tablet combinations
- Visualizations using `matplotlib` and `seaborn` (bar charts, horizontal frequency charts, etc.)

## 🧠 Models & Approaches

Several approaches were built and compared:

1. **Rule-based Search & Recommendation**
   - Filter/search functions by disease, treatment, and/or tablet
   - Frequency-based recommendation of top combinations

2. **TF-IDF + Cosine Similarity**
   - Combines Disease, Treatment, and Tablet Name into a text feature
   - Uses `TfidfVectorizer` to vectorize combined features
   - Recommends similar records using cosine similarity

3. **TF-IDF + K-Nearest Neighbors (KNN)**
   - Same TF-IDF features, using `NearestNeighbors` (cosine metric) for recommendations

4. **Random Forest Classifier**
   - Predicts the `Treatment` for a given `Disease` and `Tablet Name`
   - Includes prediction probabilities for all treatment classes
   - Achieved **64% accuracy** on the test set

### Model Evaluation

The recommendation models were evaluated using **Top-5 Accuracy** across different input combinations (Disease only, Treatment only, Tablet only, and combinations thereof), achieving up to **100% Top-5 Accuracy** for the TF-IDF + Cosine Similarity approach on certain input types.

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **numpy** – data manipulation
- **matplotlib**, **seaborn** – visualization
- **scikit-learn** – TF-IDF, Cosine Similarity, KNN, Random Forest, model evaluation
- **joblib** – model persistence

## 📁 Project Structure

├── medical.ipynb # Main Jupyter Notebook with EDA + models
├── medicine_dataset.csv # Raw dataset (not included in repo)
├── medicine_summary.csv # Cleaned/aggregated summary data (generated)
├── medicine_treatment_rf_model.pkl # Trained Random Forest model (generated)
└── README.md


## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Usage

1. Clone this repository
```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
```
2. Place `medicine_dataset.csv` in the project directory
3. Open and run the notebook
```bash
   jupyter notebook medical.ipynb
```

### Example: Predicting Treatment

```python
import joblib
import pandas as pd

# Load the trained model
rf_model = joblib.load('medicine_treatment_rf_model.pkl')

# Predict treatment for a given disease and tablet
new_data = pd.DataFrame({
    'Disease': ['Fever'],
    'Tablet Name': ['Paracetamol']
})

prediction = rf_model.predict(new_data)[0]
print("Predicted Treatment:", prediction)
```

## 📈 Results

| Model | Metric | Score |
|---|---|---|
| TF-IDF + Cosine Similarity | Top-5 Accuracy | Up to 100% |
| TF-IDF + KNN | Top-5 Accuracy | Comparable to Cosine Similarity |
| Random Forest Classifier | Accuracy | 64% |

## 🔮 Future Improvements

- Incorporate additional features (age group, purity, price) into the prediction models
- Hyperparameter tuning for the Random Forest classifier
- Build a simple web app / API (Flask or Streamlit) for interactive medicine search and recommendations
- Handle class imbalance for rarer treatment categories

## 📄 License

This project is for educational purposes. Please verify any medical information with a qualified healthcare professional — this project is **not** a substitute for professional medical advice.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
