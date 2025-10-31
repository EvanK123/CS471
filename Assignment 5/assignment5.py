import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

matplotlib.use('Agg')

training_data = pd.read_csv("training.csv", header=None, usecols=[19,23], names=['Time','Current'])
test_data = pd.read_csv("test.csv", header=None, usecols=[0, 4], names=['Time','Current'])

print(f"Training data columns: {training_data.columns}")
print(f"Test data columns: {test_data.columns}")

# filter data to relevant time ranges
training_data = training_data[training_data['Time'] <= 5.4]
test_data = test_data[test_data['Time'] <= 2.4]

print(f"\nTraining data size after filtering: {len(training_data)}")
print(f"Test data size after filtering: {len(test_data)}")

# define segmenting and labeling function
def segment_labeling(data, window, overlap, time1, time2):
    index = 0
    windolap = math.floor(window * overlap)
    labels_df = pd.DataFrame(columns=['label'])
    time_series = []

    while (index + window) < len(data):
        segment = data.iloc[index : (index+window)]
        
        # check if any time in segment falls within fault window
        if any((time1 <= t <= time2) for t in segment['Time']):
            label = 'oscillation'
        else:
            label = 'normal'

        time_series.append(segment['Current'].values) 
        labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)
        index += window - windolap

    return time_series, labels_df

# segment the data
window = 200
overlap = 0.75

train_X, train_y = segment_labeling(training_data, window, overlap, 5.1, 5.4)
test_X, test_y = segment_labeling(test_data, window, overlap, 2.1, 2.4)

# convert to numpy arrays
X_train = np.array(train_X)
X_test = np.array(test_X)

print(f"\nTraining segments: {X_train.shape}")
print(f"Test segments: {X_test.shape}")

# convert labels to numerical values
le = LabelEncoder()
y_train = le.fit_transform(train_y['label'])
y_test = le.transform(test_y['label'])

print("\nLabel mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

print(f"\nTraining set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# flatten the time series data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened training data shape: {X_train_flat.shape}")
print(f"Flattened test data shape: {X_test_flat.shape}")

# standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)


# fit initial Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# initial evaluation
y_pred = rf_clf.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Test Accuracy: {initial_accuracy:.4f}")

# hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] 
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1 
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

results = grid_search.cv_results_
n_estimators_values = sorted(set(params['n_estimators'] for params in results['params']))

# get best scores for each n_estimators value (using best other parameters)
best_n_estimator_scores = []
for n in n_estimators_values:
    # find indices with this n_estimators value
    indices = [i for i, params in enumerate(results['params']) if params['n_estimators'] == n]
    if indices:
        best_score = max(results['mean_test_score'][i] for i in indices)
        best_n_estimator_scores.append(best_score)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, best_n_estimator_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Estimators')
plt.ylabel('Best Cross-validation Accuracy')
plt.title('Random Forest Performance vs Number of Estimators')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.close()

# get best model
best_rf = grid_search.best_estimator_

y_pred_best = best_rf.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Test Accuracy: {final_accuracy:.4f}")

# classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)

# plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.colorbar()
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=12)

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# feature importance - first 20 features
feature_importance = best_rf.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(20), feature_importance[:20])
plt.xlabel('Feature Index (Time Point in Segment)')
plt.ylabel('Importance')
plt.title('Top 20 Feature Importances in Random Forest')
plt.grid(True, alpha=0.3)
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# compare initial vs tuned model performance
improvement = final_accuracy - initial_accuracy
print(f"\nImprovement after hyperparameter tuning: {improvement:.4f}")
