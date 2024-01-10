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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA on the training features
pca = PCA().fit(X_scaled)
variance_threshold = 0.95
cumulative_variance = pca.explained_variance_ratio_.cumsum()
num_components = next(x[0] for x in enumerate(cumulative_variance) if x[1] > variance_threshold) + 1
pca_18 = PCA(n_components=num_components)
X_pca_18 = pca_18.fit_transform(X_scaled)

# Create a DataFrame of the PCA results
pca_df = pd.DataFrame(X_pca_18, columns=[f'PC{i+1}' for i in range(num_components)])
pca_df['loan_status'] = y

# Splitting the PCA-transformed data into training and testing sets
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(pca_df.drop('loan_status', axis=1),
                                                                    pca_df['loan_status'], 
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
joblib.dump(pca_18, 'pca.joblib', compress=3)
joblib.dump(onehot_encoder, 'onehot_encoder.joblib', compress=3)
joblib.dump(label_encoder, 'label_encoder.joblib', compress=3)