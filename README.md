# Datheon_Machine_Learning_-_Data_Science_Internship1
Machine learning and data science internship task for Datheon Intership.
Multi-Class Classification Benchmark: Iris / Wine
📌 Project Overview

This project implements a multi-class classification benchmark using two machine learning algorithms: Logistic Regression and Decision Tree Classifier.

The project uses the built-in Iris and Wine datasets from Scikit-learn to train, evaluate, and compare model performance using multiple classification metrics and visualizations.

The benchmark supports both datasets, allowing users to switch between Iris and Wine by changing a single configuration variable.

🎯 Objectives
Explore and analyze the Iris and Wine datasets.
Perform exploratory data analysis (EDA) using visualizations.
Split the dataset into training and testing sets.
Train Logistic Regression and Decision Tree models.
Evaluate models using classification metrics.
Compare model performance using charts and confusion matrices.
Visualize decision boundaries using two selected features.
Identify the best-performing model based on Macro F1-score.
Export benchmark results to a CSV file.
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
📂 Datasets
1. Iris Dataset

The Iris dataset contains measurements of iris flowers and classifies them into three species.

Features:

Sepal length
Sepal width
Petal length
Petal width

Target classes:

Setosa
Versicolor
Virginica
2. Wine Dataset

The Wine dataset contains chemical analysis measurements of wines from three different cultivars.

It includes 13 numerical features, such as alcohol, malic acid, ash, magnesium, and color intensity.

The target variable represents three wine classes.

Note: The project uses the Iris dataset by default. To run the benchmark on the Wine dataset, change DATASET_NAME = "iris" to DATASET_NAME = "wine".

🤖 Machine Learning Models
Logistic Regression
Uses feature scaling with StandardScaler.
Supports multi-class classification.
Uses a maximum of 2,000 iterations for model convergence.
Decision Tree Classifier
Learns decision rules from the training data.
Supports multi-class classification.
Uses a fixed random state for reproducibility.

Both models are implemented using Scikit-learn pipelines.

📊 Exploratory Data Analysis

The project performs the following analyses:

Dataset overview and first five rows
Dataset information and missing-value checks
Class distribution analysis
Statistical summary of numerical features
Feature pairplot
Feature correlation heatmap
📈 Model Evaluation Metrics

The models are evaluated using the following metrics:

Metric	Description
Accuracy	Overall proportion of correct predictions
Macro Precision	Average precision across all classes
Macro Recall	Average recall across all classes
Macro F1-score	Average F1-score across all classes

Macro averaging gives equal weight to each class, making it useful for evaluating multi-class classification performance.

The project also generates classification reports and confusion matrices to analyze class-wise predictions.

📉 Visualizations

The project generates the following visualizations:

Class distribution plot
Feature pairplot
Feature correlation heatmap
Model metric comparison bar chart
Confusion matrix heatmaps
Two-dimensional decision boundary plots

The decision boundary plots use the first two features for visualization, while the main benchmark models use all available features.

⚙️ Installation and Setup
1. Clone the repository
bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git

2. Navigate to the project directory
bash
cd YOUR-REPOSITORY

3. Install the required libraries
bash
pip install numpy pandas matplotlib seaborn scikit-learn

4. Run the Python script
bash
python classification_benchmark.py


Replace classification_benchmark.py with your actual Python filename if it is different.

▶️ How to Use

Open the Python script and select the dataset:

python
DATASET_NAME = "iris"


For the Wine dataset, use:

python
DATASET_NAME = "wine"


Run the script to train both models, evaluate their performance, generate visualizations, and save the benchmark results.

The dataset is split into 80% training data and 20% testing data, using stratified sampling and a fixed random state of 42.

📁 Project Structure
text
classification-benchmark/
│
├── classification_benchmark.py
├── iris_classification_benchmark.csv
├── wine_classification_benchmark.csv
├── README.md
└── requirements.txt


The CSV files will be generated when the corresponding dataset benchmark is executed. Include only the files that actually exist in your repository.

📄 Output

The program produces:

Classification reports for both models
A model comparison table
Metric comparison visualizations
Confusion matrix heatmaps
Two-dimensional decision boundary plots
A CSV file containing the evaluation metrics

The CSV output includes accuracy, macro precision, macro recall, and macro F1-score for each model.

🏆 Best Model Selection

The best-performing model is selected automatically based on the highest Macro F1-score.

The final output displays the selected model's name and its evaluation metrics.

Actual benchmark scores may vary depending on the selected dataset and configuration.

🔑 Key Learnings
Understanding multi-class classification
Comparing linear and tree-based machine learning models
Applying feature scaling using Scikit-learn pipelines
Performing exploratory data analysis
Evaluating models with classification metrics
Interpreting confusion matrices and decision boundaries
Exporting model evaluation results to CSV
🚀 Future Improvements
Add Random Forest, Support Vector Machine, and K-Nearest Neighbors classifiers.
Perform hyperparameter tuning using GridSearchCV.
Add cross-validation for more robust model evaluation.
Save trained models for future predictions.
Build an interactive dashboard for model comparison.
👨‍💻 Author

Your Name

GitHub: https://github.com/YOUR-USERNAME

📜 License

This project is intended for educational and machine learning practice purposes.

the next project ,
California Housing Price Prediction
📌 Project Overview

This project focuses on predicting median house values using the California Housing dataset and comparing three regression models: Ordinary Least Squares (OLS), Ridge Regression, and Lasso Regression.

The project includes exploratory data analysis (EDA), data preprocessing, polynomial interaction features, hyperparameter tuning using GridSearchCV, model evaluation, and residual analysis.

The best-performing model is selected based on the lowest Root Mean Squared Error (RMSE) on the test dataset.

🎯 Objectives
Analyze the California Housing dataset.
Perform exploratory data analysis using statistical summaries and visualizations.
Preprocess data using median imputation and feature scaling.
Generate pairwise interaction features.
Train OLS, Ridge, and Lasso regression models.
Optimize regularization parameters using cross-validation.
Evaluate models using RMSE, MAE, and R².
Visualize predicted versus actual values and residual distributions.
Examine Lasso coefficients for feature selection.
Export model comparison results to a CSV file.
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
📊 Dataset

California Housing Dataset

The dataset is available through Scikit-learn's fetch_california_housing() function.

It contains census-block-group-level information about housing values and demographic characteristics in California.

Features

The dataset includes eight input features:

Feature	Description
MedInc	Median income
HouseAge	Median house age
AveRooms	Average number of rooms
AveBedrms	Average number of bedrooms
Population	Population
AveOccup	Average household occupancy
Latitude	Geographic latitude
Longitude	Geographic longitude

Target variable: MedHouseVal — median house value, expressed in units of $100,000.

The dataset contains 20,640 observations and eight input features.

🔍 Exploratory Data Analysis

The project performs the following analyses:

Dataset shape and first five rows
Data types and missing-value checks
Descriptive statistical summaries
Distribution of median house values
Correlation heatmap
Identification of highly correlated predictor pairs

These analyses help understand feature distributions, relationships, and potential multicollinearity.

⚙️ Data Preprocessing

The preprocessing pipeline includes:

Missing-value imputation: Replaces missing numerical values with the median.
**
