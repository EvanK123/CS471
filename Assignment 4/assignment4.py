# CS471: Introduction to Artificial Intelligence 
# Assignment 4: Decision Tree Classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# load data and basic description
data = load_iris()
X, y = data.data, data.target

print("Features:", data.feature_names)
print("Target classes:", data.target_names)


# split into train (60%), validation (20%), test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

#fit baseline decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Baseline validation accuracy:", accuracy_score(y_val, clf.predict(X_val)))

# hyperparameter tuning
# tune max_depth and min_samples_split
max_depth_values = range(1, 11) # 1 to 10 depth
min_samples_split_values = range(2, 11) # min samples required to split node

# evaluation of different max_depth values
val_scores_depth = []
for d in max_depth_values:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    val_scores_depth.append(accuracy_score(y_val, model.predict(X_val)))

# evaluatation of different min_samples_split values
val_scores_split = []
for s in min_samples_split_values:
    model = DecisionTreeClassifier(min_samples_split=s, random_state=42)
    model.fit(X_train, y_train)
    val_scores_split.append(accuracy_score(y_val, model.predict(X_val)))

# plot validation performance
plt.figure(figsize=(6,4))
plt.plot(max_depth_values, val_scores_depth, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs max_depth")
plt.grid(True)

plt.savefig("validation_max_depth.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(min_samples_split_values, val_scores_split, marker="o", color="orange")
plt.xlabel("min_samples_split")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs min_samples_split")
plt.grid(True)

plt.savefig("validation_min_samples_split.png")
plt.close()

# choose best hyperparameters
best_depth = max_depth_values[np.argmax(val_scores_depth)]
best_split = min_samples_split_values[np.argmax(val_scores_split)]
print("\nBest max_depth:", best_depth)
print("Best min_samples_split:", best_split)

# retrain on train+val with best hyperparameters
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])

final_model = DecisionTreeClassifier(
    max_depth=best_depth, min_samples_split=best_split, random_state=42
)
final_model.fit(X_train_val, y_train_val)

# evaluate final model on the test set
y_pred = final_model.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# visualize and interpret the decision tree
plt.figure(figsize=(12,8))
plot_tree(
    final_model, 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    filled=True, 
    rounded=True
)
plt.savefig("decision_tree.png")
plt.close()