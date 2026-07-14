from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the Logistic Regression model for multi-class classification
# Using the scaled classification features and target
log_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
log_reg_model.fit(X_train_clf_scaled, y_train_clf)

# Make predictions on the scaled test set
y_pred_log_reg = log_reg_model.predict(X_test_clf_scaled)

print("--- Logistic Regression Classification Evaluation ---")
print("Classification Accuracy:")
print(accuracy_score(y_test_clf, y_pred_log_reg))

print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_log_reg))
