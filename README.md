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


