# Insurance_Fraud_Detection_Project

### Project Overview

The project aimed to develop an advanced predictive model for Travelers Insurance Company to identify first-party physical damage fraudulence. By leveraging machine learning techniques, the model achieved an accuracy of 85% and an F1 score of 78%. The approach involved applying a range of data pre-processing techniques to prepare the dataset and selecting optimal hyperparameters to enhance the model's performance. The model's efficacy was demonstrated through a comprehensive classification report and confusion matrix analysis for both training and test partitions, ensuring robust fraud detection capabilities.

### Tools

- Programming Language: Python
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
- Environment: Google Colab

### Data Cleaning

- Data Source: The dataset contained 2 GB of data with features relevant to fraud detection.
- Handling Missing Values:
  - Identified missing values in the dataset using isnull() method.
  - Imputed missing values using appropriate techniques (e.g., mean, median, mode).
- Outlier Detection:
  - Used box plots and statistical methods to detect and handle outliers.
- Data Transformation:
  - Encoded categorical variables using techniques such as one-hot encoding.
  - Normalized numerical features to ensure they are on a similar scale.
 
### Exploratory Data Analysis (EDA)
- Visualizations:
  - Used bar plots, histograms, and box plots to visualize the distribution of features.
  - Analyzed the correlation between different features and the target variable (fraud).
- Group Analysis:
  - Grouped data by key features such as gender, marital status, and claim day of the week to identify patterns in fraudulent activities.
 
### Data Analysis
- Feature Engineering:
  - Created new features from existing ones to enhance model performance.
- Model Selection:
  - Evaluated multiple machine learning models including Decision Tree, Random Forest, and Gradient Boosting Classifier.
- Model Tuning:
  - Performed hyperparameter tuning using techniques like RandomizedSearchCV to optimize model performance.
- Model Training:
  - Split data into training and testing sets.
  - Trained models on the training set and evaluated them on the testing set.
- Performance Metrics:
  - Accuracy
  - F1 Score
  - Precision and Recall
 
```python
// Group the data by the "gender" feature and calculate summary statistics for each group
grouped_data = data_train.groupby('gender').agg({'fraud': ['count']})

// Print the grouped data
print('Fraud claims grouped by gender:\n', grouped_data)

// Concatenate X_train and y_train into a single DataFrame
data_train = pd.concat([X_train, y_train], axis=1)

// Group the data by the "marital status" feature and calculate summary statistics for each group
grouped_data = data_train.groupby('marital_status').agg({'fraud': ['count']})

// Print the grouped data
print('Fraud claims grouped by marital status:\n', grouped_data)

// Concatenate X_train and y_train into a single DataFrame
data_train = pd.concat([X_train, y_train], axis=1)

// Group the data by the "claim day of the week" feature and calculate summary statistics for each group
grouped_data = data_train.groupby('claim_day_of_week').agg({'fraud': ['count']})

// Print the grouped data
print('Fraud claims grouped by Day of the Week:\n', grouped_data)

// Check for the presence of missing values in X_train
print(X_train.isnull().sum())
// Note that there are 305 missing values in X_train (this also includes annual_income replaced -1 values and age_of_driver NaN coded values)

null_columns_X_train = X_train.columns[X_train.isnull().any()]
print(null_columns_X_train)

// Check for the presence of missing values in X_test
print(X_test.isnull().sum())
// There are 141 missing values in X_train

null_columns_X_test = X_test.columns[X_test.isnull().any()]
print(null_columns_X_test)

// Review the record with missing values before imputation
X_train.loc[15994]

// Print the row with missing values
row_with_missing_values = X_train.loc[15994]

// Create a DataFrame with the row
df_row = pd.DataFrame([row_with_missing_values]).T

// Create a Styler object for the DataFrame
styled_df = df_row.style.apply(lambda x: ['background-color: red' if pd.isnull(v) else '' for v in x])

// Display the styled DataFrame
styled_df

// Imputing the missing values in X_train

// Turn off empty subplots
for i in range(len(features), nrows*ncols):
    axes.flatten()[i].axis('off')

// Create a dictionary to store the ICE (Individual Conditional Expectation) plots for each feature using the ice() function from some library, with predictions made by DTC.predict()
train_ice_dfs = {feat: ice(data=train_X_df[0:1000], column=feat, predict=RFC.predict) for feat in Features}

// Plot the ICE grids using plot_ice_grid() function
plot_ice_grid(train_ice_dfs, train_X_df, Features,
              ax_ylabel='Predicted fraud value', alpha=0.5, plot_pdp=True,
              pdp_kwargs={'c': 'blue', 'linewidth': 2},
              linewidth=0.2, c='black')
plt.xlim([0, 0.8])
plt.ylim([0, 1])
plt.suptitle('ICE plots (training data): RFC model')
plt.subplots_adjust(top=0.89)
plt.tight_layout()

// Perform randomized search cross-validation for Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV

// Create a Random Forest Classifier
rfc = RandomForestClassifier()

// Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 5, 10, None],
    'min_samples_split': [1, 2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 5, 10],
}

// Perform randomized search cross-validation
random_search = RandomizedSearchCV(rfc, param_distributions=param_grid, cv=5, n_iter=20)
random_search.fit(X_train, y_res)

// Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

// Train the Gradient Boosting Classifier
GBC = GradientBoostingClassifier(learning_rate=0.02, max_depth=5, n_estimators=300)
GBC = GBC.fit(X_train, y_res)

// Store the predictions for the training set
train_preds_GBC = GBC.predict(X_train)

// Store the predictions for the test set
test_preds_GBC = GBC.predict(X_test)

// Generate and print the classification report for the training set
trainReport_GBC = classification_report(y_res, train_preds_GBC)
print(trainReport_GBC)

// Generate and print the classification report for the test set
testReport_GBC = classification_report(y_test, test_preds_GBC)
print(testReport_GBC)
```


 ### Results/Findings
- Model Performance:
  - Achieved an accuracy of 88% with the Random Forest Classifier.
  - Obtained a weighted F1 score of 78%.
- Key Insights:
  - Identified critical fraud indicators through feature importance analysis.
  - Certain demographics and claim patterns were more prone to fraudulent activities.
 
### Recommendations
- Improving Fraud Detection:
  - Implement the predictive model into the existing fraud detection system.
  - Regularly update the model with new data to maintain its accuracy and relevance.
- Risk Management:
  - Use analytical findings to enhance risk management strategies.
  - Educate stakeholders on the key fraud indicators to improve vigilance.
 
### Limitations
- Data Limitations:
  - The dataset used may not cover all possible fraud scenarios, leading to potential blind spots in the model.
- Model Limitations:
  - The model may have limitations in generalizing to new, unseen data.
  - Potential overfitting despite cross-validation and regularization efforts.


