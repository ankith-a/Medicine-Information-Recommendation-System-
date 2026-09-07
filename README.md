# 💊 Medicine Information & Recommendation System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A complete Data Science and Machine Learning project that provides a **medicine information and recommendation system** using Disease, Tablet Name, and Treatment relationships.

The project demonstrates the complete workflow of working with a medicine dataset, including **data cleaning, exploratory data analysis (EDA), relationship analysis, data preprocessing, machine learning, model saving, and Streamlit web application development**.

The system allows users to search medicine information using different combinations of **Disease, Tablet Name, and Treatment** and retrieve corresponding medicine information from the dataset.

---

# 📌 Project Objective

The objective of this project is to build an interactive system that helps users explore relationships between:

- Disease
- Treatment
- Tablet Name

The system can retrieve corresponding information based on one, two, or all three input fields.

It also provides additional medicine details such as:

- Dosage
- Time Taken
- Effect Time Duration
- Age Group
- Company
- Purity
- Side Effects
- Price

The project also includes a **Random Forest model** for machine learning-based treatment prediction.

> ⚠️ This project is intended for informational and educational purposes only. It is not a medical prescription or a substitute for professional medical advice.

---

# 📊 Dataset

The project uses a medicine dataset containing **100,000 records and 11 columns**.

### Dataset Includes

- Disease
- Treatment
- Tablet Name
- Dosage
- Time Taken
- Effect Time Duration
- Which Age
- Company
- Purity
- Side Effects
- Price

### Dataset Statistics

| Feature | Unique Values |
|---------|---------------|
| Disease | 25 |
| Treatment | 26 |
| Tablet Name | 57 |
| Dosage | 14 |
| Time Taken | 10 |
| Effect Time Duration | 8 |
| Which Age | 6 |
| Company | 12 |
| Purity | 50 |
| Side Effects | 2380 |
| Price | 58668 |

The dataset was cleaned and prepared before analysis and application development.

Data preprocessing included:

- Removing unnecessary whitespace
- Cleaning column names
- Converting Purity values into numerical format
- Converting Price values into numerical format
- Checking missing values
- Checking duplicate records
- Preparing categorical data for analysis and modeling

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Cleaning & Analysis |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Scikit-Learn | Machine Learning |
| Joblib | Model Saving & Loading |
| Streamlit | Web Application |
| Jupyter Notebook | Data Science Development |
| PyCharm | Application Development |

---

# 🔄 Project Workflow

### 1️⃣ Data Collection

- Load the medicine dataset
- Inspect dataset structure
- Check columns and data types
- Analyze dataset size
- Identify categorical and numerical features

---

### 2️⃣ Data Cleaning

The dataset was cleaned before performing analysis.

Steps included:

- Cleaning column names
- Removing leading and trailing spaces
- Checking missing values
- Checking duplicate records
- Cleaning Purity values
- Cleaning Price values
- Preparing data for analysis

Example:

```python
data.columns = data.columns.str.strip()

text_columns = data.select_dtypes(
    include='object'
).columns

for col in text_columns:
    data[col] = data[col].str.strip()
