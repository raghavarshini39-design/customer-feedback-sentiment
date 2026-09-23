# Customer Feedback Sentiment

**Intern ID:** CITS8924  
**Full Name:** CHINTAGINJALA RAGHA VARSHINI  
**No. of Weeks:** 6 Weeks  
**Project Name:** Customer Feedback Sentiment Analysis

## CODTECH Task 3
This project analyzes customer feedback and classifies each review as **Positive, Neutral, or Negative**.

## Project Scope
- Data cleaning and preprocessing
- Sentiment distribution analysis
- Visualization
- Sentiment prediction using TF-IDF and Logistic Regression
- Model evaluation
- Output files suitable for a dashboard

## Dataset
The included `data/customer_feedback.csv` is a dummy dataset created for project implementation. CODTECH allows dummy datasets or public datasets for Data Science & Analytics projects.

## Tools & Technologies
- Python
- Pandas
- Matplotlib
- Scikit-learn
- VS Code / Jupyter Notebook
- GitHub
- Power BI (optional dashboard)

## Folder Structure
```text
Customer_Feedback_Sentiment/
├── data/
│   └── customer_feedback.csv
├── src/
│   └── sentiment_analysis.py
├── outputs/
│   └── generated charts, predictions and model results
├── docs/
│   └── project_documentation.md
├── requirements.txt
└── README.md
```

## How to Run
1. Install Python 3.x.
2. Open the project folder in VS Code.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run:
   `python src/sentiment_analysis.py`
5. Check the `outputs` folder for charts, predictions and model results.

## Dashboard
Import `outputs/predictions.csv` into Power BI and create:
- Total feedback count
- Positive/Neutral/Negative counts
- Sentiment percentage
- Sentiment column chart
- A table of feedback and predicted sentiment
