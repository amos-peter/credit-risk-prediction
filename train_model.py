# Import the required packages
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('data_cleaned.zip', compression='zip')
data.drop(columns=['Unnamed: 0'], inplace=True)

# List of numerical columns for distribution analysis
numerical_columns = ['person_age', 'person_income', 'person_emp_length', 
                     'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                     'cb_person_cred_hist_length']

# Function to apply winsorization based on Z-score
def winsorize_by_zscore(data, column, lower_bound=-3, upper_bound=3):
    z_scores = zscore(data[column])
    lower_limit = data[column][z_scores > lower_bound].min()
    upper_limit = data[column][z_scores < upper_bound].max()
    data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
    return data

# Applying winsorization based on Z-score to each numerical column
winsorized_data_zscore = data.copy()
for col in numerical_columns:
    winsorized_data_zscore = winsorize_by_zscore(winsorized_data_zscore, col)

# Initialize and fit encoders
onehot_encoder = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()
label_encoder.fit(winsorized_data_zscore['cb_person_default_on_file'])
onehot_encoded_data = onehot_encoder.fit_transform(winsorized_data_zscore[['person_home_ownership', 'loan_intent', 'loan_grade']])
label_encoded_data = label_encoder.transform(winsorized_data_zscore['cb_person_default_on_file'])

# Create DataFrames for encoded data
onehot_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(['person_home_ownership', 'loan_intent', 'loan_grade']))
label_df = pd.DataFrame(label_encoded_data, columns=['cb_person_default_on_file_encoded'])

# Concatenate the encoded data with the original dataset
encoded_data = pd.concat([winsorized_data_zscore.drop(columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']), 
                          onehot_df, label_df], axis=1)

# Feature Scaling
X = encoded_data.drop('loan_status', axis=1)  # Features
y = encoded_data['loan_status']               # Target

# Initializing the StandardScaler Z-Score Scalling
scaler = StandardScaler()

# Applying Standard Scaling to the features
X_scaled = scaler.fit_transform(X)

# Creating a DataFrame from the scaled features
scaled_data = pd.DataFrame(X_scaled, columns=X.columns)
scaled_data['loan_status'] = y  # Adding the target variable back correctly

# lasso
# Separate features and target
X = scaled_data.drop('loan_status', axis=1)  # Ensure 'data' is your original DataFrame
y = scaled_data['loan_status']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply Lasso Regression
lasso = Lasso(alpha=0.01)  # Adjust alpha as needed
lasso.fit(X_train, y_train)

# Evaluate the model
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Extracting the coefficients
coefficients = lasso.coef_
selected_features = X.columns[(coefficients != 0)]

lasso_df = scaled_data[list(selected_features)+['loan_status']]

# Splitting the PCA-transformed data into training and testing sets
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(lasso_df.drop('loan_status', axis=1),
                                                                    lasso_df['loan_status'], 
                                                                    test_size=0.2, 
                                                                    random_state=42)

# Handling imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_pca_train_smote, y_pca_train_smote = smote.fit_resample(X_pca_train, y_pca_train)

# Training the Random Forest model with reduced complexity
random_forest = RandomForestClassifier(n_estimators=50, random_state=42)
random_forest.fit(X_pca_train_smote, y_pca_train_smote.values.ravel())

# Model evaluation
y_pred_rf = random_forest.predict(X_pca_test)
accuracy_rf = accuracy_score(y_pca_test, y_pred_rf)
report_rf = classification_report(y_pca_test, y_pred_rf, output_dict=True)

# Displaying the accuracy and the classification report
print("Accuracy of Random Forest: {:.2f}%".format(accuracy_rf * 100))
print("Classification Report for Random Forest:")
print(pd.DataFrame(report_rf).transpose())

# Save the trained Random Forest model and other components using joblib for better compression
joblib.dump(random_forest, 'loan_status_model.joblib', compress=3)
joblib.dump(scaler, 'scaler.joblib', compress=3)
joblib.dump(lasso, 'lasso.joblib', compress=3)
joblib.dump(onehot_encoder, 'onehot_encoder.joblib', compress=3)
joblib.dump(label_encoder, 'label_encoder.joblib', compress=3)