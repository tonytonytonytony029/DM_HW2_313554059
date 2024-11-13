import pandas as pd

# 生成Kaggle提交檔案
def create_submission_file(model, X_test, output_file='testing_result.csv'):
    predictions = model.predict(X_test)
    submission = pd.DataFrame({'Id': X_test.index, 'Prediction': predictions})
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")
