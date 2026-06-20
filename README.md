# 🚢 Titanic Survival Analysis (EDA Project)

## Project Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic dataset to identify the factors that influenced passenger survival.

The analysis includes data cleaning, feature engineering, statistical summaries, visualizations, and key insights.

---

## Dataset

* Titanic Dataset
* 891 passenger records
* 12 original features

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Project Workflow

### 1. Data Loading

* Loaded the Titanic dataset
* Explored dataset structure and data types

### 2. Data Cleaning

* Filled missing Age values
* Filled missing Embarked values
* Replaced missing Cabin values with "Unknown"

### 3. Feature Engineering

Created new features:

* FamilySize
* IsAlone
* AgeGroup
* FareGroup
* Sex_encoded

### 4. Exploratory Data Analysis (EDA)

Analyzed:

* Survival distribution
* Gender impact on survival
* Passenger class impact
* Age and survival
* Fare and survival
* Family size impact
* Correlation between variables

### 5. Visualizations

Created 8 visualizations:

1. Overall Survival Overview
2. Survival by Gender
3. Survival by Passenger Class
4. Age Analysis
5. Fare & Embarkation Analysis
6. Family Size Analysis
7. Correlation Heatmap
8. Gender × Class Analysis

---

## Key Insights

* Overall survival rate was 38.4%.
* Female passengers had significantly higher survival rates than males.
* First-class passengers survived more often than third-class passengers.
* Children had better survival rates than adults.
* Higher ticket fares were associated with higher survival rates.
* Small families had better survival chances.
* Cherbourg passengers had the highest survival rate.
* Gender and passenger class were the strongest factors affecting survival.

---
## How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn
```

Run the project:

```bash
python titanic_eda.py
```

---

## Learning Outcomes

Through this project, I gained practical experience in:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Data Visualization
* Correlation Analysis
* Insight Generation

---

## Author

Titanic Survival Analysis Project completed using Python for Exploratory Data Analysis and Data Visualization.
