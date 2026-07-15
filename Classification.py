import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create a categorical target for classification by binning 'climate_risk_index'
df_classification = df.copy() # Work on a copy to not affect previous regression analysis

# Define bins for climate_risk_index based on quantiles
bins = df_classification['climate_risk_index'].quantile([0, 0.33, 0.66, 1]).tolist()
labels = ['Low', 'Medium', 'High']

# Handle cases where quantiles might not be distinct (unlikely for this dataset size, but good practice)
if len(set(bins)) < 4: # If not enough distinct quantiles, adjust bins to include max value
    min_val = df_classification['climate_risk_index'].min()
    max_val = df_classification['climate_risk_index'].max()
    bins = [min_val, min_val + (max_val - min_val)/3, min_val + 2*(max_val - min_val)/3, max_val]
    bins[-1] = bins[-1] + 1e-6 # Add a small epsilon to include the absolute max value

df_classification['climate_risk_category'] = pd.cut(df_classification['climate_risk_index'], bins=bins, labels=labels, include_lowest=True)

# Define X_clf (features) and y_clf (new categorical target) for classification
# Exclude the original continuous target and the new categorical target from features
X_clf = df_classification.drop(['climate_risk_index', 'climate_risk_category'], axis=1)
y_clf = df_classification['climate_risk_category']

# Ensure all features in X_clf are numerical. 'country' was already encoded in previous steps.
# If there were any other object/category columns, they would need encoding here.
for col in X_clf.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_clf[col] = le.fit_transform(X_clf[col])

print("New target variable 'climate_risk_category' created:")
display(y_clf.head())
print("\nDistribution of 'climate_risk_category':")
display(y_clf.value_counts())
