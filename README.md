# Abstract
Heart failure is an increasingly common and potentially fatal condition that occurs when a person's heart is unable to pump or fill with enough blood to support the body. In this study, I examined a data set of 300 heart failure patients and their associated lifestyle and biological data. The purpose of this study is to 1: determine the biomarkers with the strongest predictive power and 2: create a robust predictive model to determine patient survivability. First, I performed an exploratory data analysis of the patient data. Using these insights, I trained several classifiers to determine survivability. Through this process, I find ejection fraction, serum creatinine, and age are the 3 most effective predictors of a heart failure patient's survival. 

Created using:
dependencies = [
    "great-tables>=0.18.0",
    "jinja2>=3.1.6",
    "marimo>=0.14.17",
    "matplotlib>=3.10.5",
    "mlxtend>=0.23.4",
    "numpy>=2.3.2",
    "pandas>=2.3.1",
    "polars>=1.32.3",
    "pyarrow>=21.0.0",
    "ruff>=0.12.9",
    "scikit-learn>=1.7.1",
    "selenium>=4.35.0",
    "tabulate>=0.9.0",
    "xgboost>=3.0.4",
]
