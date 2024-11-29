# **Food Hazard Detection Challenge: SemEval 2025 Task 9**

This repository contains the solution for **SemEval 2025 Task 9: The Food Hazard Detection Challenge**. This task focuses on classifying food-incident reports collected from the web, using explainable classification models to predict food hazards and product categories. These algorithms aim to support automated crawlers in detecting food-related issues from online sources such as social media, potentially helping prevent foodborne illnesses.

## **Table of Contents**

- [Introduction](#introduction)
- [Task Description](#task-description)
- [Data Overview](#data-overview)
- [Solution Overview](#solution-overview)
- [System Components](#system-components)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Development](#model-development)
- [Benchmark Analysis](#benchmark-analysis)
  - [Subtask 1: Hazard and Product Category Prediction](#subtask-1-hazard-and-product-category-prediction)
  - [Subtask 2: Hazard and Product Vector Prediction](#subtask-2-hazard-and-product-vector-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run the Code](#how-to-run-the-code)
- [References](#references)

## **Introduction**

The **Food Hazard Detection Challenge** evaluates explainable classification models for food-incident reports. This challenge combines two tasks:

- **Subtask 1 (ST1)**: Predict the **type of hazard** and **product category** from the titles or text of food recall reports.
- **Subtask 2 (ST2)**: Predict the **exact hazard** and **product vector** from the titles or text of food recall reports.

The repository includes both basic and advanced machine learning models to benchmark the performance for hazard detection using short texts (**titles**) and long texts (**full text reports**).

## **Task Description**

The **Food Hazard Detection Challenge** involves creating classification models to predict food hazards and products from incident reports. This involves:

- Classifying the **"hazard-category"** and **"product-category"** from food recall titles and full texts.
- Predicting specific **hazard** and **product vectors**.

The main goals are:
- **Explainability**: Develop interpretable models that support human experts in assessing food risks.
- **High Performance**: Achieve robust classification scores based on the **macro F1-Score**, focusing primarily on hazard labels.

## **Data Overview**

- **Training Data**: Labeled dataset containing 5,082 samples. Features include:
  - **Title**: Short description of the food recall.
  - **Text**: Full description of the food recall.
  - **Labels**: `hazard-category`, `product-category`, `hazard`, `product`.

- **Validation Data**: Unlabeled dataset with 565 samples for performance assessment.

- **Test Data**: Unlabeled dataset with 997 samples for evaluation during the challenge.

The dataset contains **22 product categories** and **10 hazard categories** sorted into specific **product vectors** (e.g., "ice cream", "meat products") and **hazard vectors** (e.g., "listeria monocytogenes", "milk allergy").

## **Solution Overview**

The solution involves:
- Benchmarking two models (**Logistic Regression** as a baseline and **CatBoost Classifier** as an advanced model).
- Performing separate analyses on **titles** and **full texts**.
- Using TF-IDF vectorization to convert text data into numerical features.
- Evaluating models using the **macro F1-score** to handle the imbalanced class distribution.

## **System Components**

### **Data Preprocessing**

1. **Text Normalization**: Text was normalized by converting to lowercase, removing special characters and numbers.
2. **Stopword Removal**: Common stopwords were removed.
3. **Lemmatization**: Words were reduced to their base forms using lemmatization.

### **Feature Engineering**

1. **TF-IDF Vectorization**:
   - **Title Vectorization**: Extracted 50 features using **unigrams** and **bigrams**.
   - **Text Vectorization**: Extracted 200 features using **unigrams** and **bigrams**.

2. **Handling Class Imbalance**: Rare classes (fewer than 5 samples) were grouped into an "other" category.

### **Model Development**

1. **Logistic Regression**: Baseline model trained using class weighting to address imbalance.
2. **CatBoost Classifier**: Advanced model used for both Subtasks 1 and 2, leveraging CatBoost's ability to handle categorical data effectively.

## **Benchmark Analysis**

### **Subtask 1: Hazard and Product Category Prediction**
- Models used: **Logistic Regression** and **CatBoost Classifier**.
- The **CatBoost Classifier** outperformed **Logistic Regression** with an F1-score of **0.460** for hazard-category prediction on text data.
- Titles were less informative, with both models struggling to achieve high accuracy due to the limited context in short texts.

### **Subtask 2: Hazard and Product Vector Prediction**
- Models used: **Logistic Regression** and **CatBoost Classifier**.
- **CatBoost** performed slightly better than the baseline, but results were generally low due to data complexity.
- Titles again performed poorly compared to full texts, reflecting the importance of detailed descriptions for accurate classification.

## **Evaluation Metrics**

- **Macro F1-Score** was used to evaluate performance, focusing on balanced performance across all classes.
- The scoring function provided by the organizers was used to calculate final scores based on hazard and product predictions.

```python
from sklearn.metrics import f1_score

def compute_score(hazards_true, products_true, hazards_pred, products_pred):
    f1_hazards = f1_score(hazards_true, hazards_pred, average='macro')
    f1_products = f1_score(
        products_true[hazards_pred == hazards_true],
        products_pred[hazards_pred == hazards_true],
        average='macro'
    )
    return (f1_hazards + f1_products) / 2
```

## **Results**

### **Combined Score Summary**

- **Titles** (Subtask 1): Combined score of **0.0989**.
- **Texts** (Subtask 1): Combined score of **0.1087**.
- **Titles** (Subtask 2): Combined score of **0.3176**.
- **Texts** (Subtask 2): Combined score of **0.3902**.

The **CatBoost Classifier** performed slightly better on text data than titles, indicating the importance of full contextual information in accurate hazard and product classification.

### **Visualization**

![Combined F1 Score Visualization for Text](images/combined_f1_score_text.png)

- **Texts** outperformed **titles** in both subtasks.
- The **hazard category** predictions were generally more accurate than **product predictions**, possibly due to greater consistency in hazard-related terminology.

## **Conclusion**

- **Text Analysis**: Full texts provide more context and thus improve model accuracy compared to titles.
- **Advanced Model Performance**: The **CatBoost Classifier** offers improvements over the baseline Logistic Regression model, especially in handling categorical features and class imbalance.
- **Challenges**: Class imbalance and underrepresented categories remain significant challenges, affecting the model's ability to generalize effectively.

**Future Directions**:
- **Data Augmentation**: Techniques like **SMOTE** could improve performance for underrepresented classes.
- **Advanced Architectures**: Exploring **transformer models** for text analysis could yield better results.

## **How to Run the Code**

1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-username/food-hazard-detection-semeval-2025.git
   cd food-hazard-detection-semeval-2025
   ```

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run Data Preprocessing**
   ```sh
   python preprocess_data.py
   ```

4. **Train and Evaluate Models**
   ```sh
   python train_model.py --model logistic_regression
   python train_model.py --model catboost
   ```

5. **Generate Predictions for Submission**
   ```sh
   python generate_predictions.py
   ```

## **References**
- **SemEval 2025 Food Hazard Detection Challenge**: [SemEval 2025 Task 9 on GitHub](https://github.com/link-to-semeval-task-9)
- **CatBoost Documentation**: [CatBoost Official Docs](https://catboost.ai/)
- **SKLearn Documentation**: [Scikit-Learn](https://scikit-learn.org/stable/)



