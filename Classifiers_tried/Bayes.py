import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths and constants
AUDIO_FOLDER = "CBU0521DD_stories"
ATTRIBUTE_FILE = "../CBU0521DD_stories_attributes.csv"
FEATURES_FILE = "../selected_features.csv"

# Load features from CSV file
features_df = pd.read_csv(FEATURES_FILE)

# Extract features and labels
features = features_df.iloc[:, :-1].values  # All columns except the last one (features)
labels = features_df['label'].values  # The last column is the label

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Set up hyperparameter grid for GridSearchCV
param_grid = {
    'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, ]  # Example parameter for GaussianNB
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=8, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_nb_classifier = grid_search.best_estimator_

# Make predictions
train_predictions = best_nb_classifier.predict(X_train)
test_predictions = best_nb_classifier.predict(X_test)

# Evaluate performance
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
report = classification_report(y_test, test_predictions)

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(report)
