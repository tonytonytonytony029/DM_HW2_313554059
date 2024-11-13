# HW2 Data Mining Lab Assignment

## Project Structure
- `data_preprocessing.py`: Handles data loading, cleaning, transformation, and balancing.
- `model_training.py`: Contains model training and evaluation functions.
- `cross_validation.py`: Runs KFold cross-validation to calculate average AUROC and F1-Score.
- `feature_importance.py`: Analyzes and visualizes feature importance.
- `kaggle_submission.py`: Generates the CSV file for Kaggle submission.
- `utils/`: Contains utility functions for data loading/saving and metric calculations.

## How to Run

1. **Data Preprocessing**:
```bash
   python data_preprocessing.py
```
2. **Model Training:**
```bash
    python model_training.py
```
3. **Cross Validation:**
```bash
    python cross_validation.py
```
4. **Feature Importance Analysis:**
```bash
    python feature_importance.py
```  
5. **Generate Kaggle Submission:**
```bash
    python kaggle_submission.
```

## Requirements
- Python 3.x
- pandas, scikit-learn, shap, imbalanced-learn, joblib, matplotlib, seaborn

## Notes
- Modify paths and parameters as necessary based on your setup.
- Ensure your data file is correctly placed or paths adjusted in data_preprocessing.py.