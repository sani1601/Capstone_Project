from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train Test Split
# -------------------------------

# Assume last column is target

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# REGRESSION
# ===============================

# Only if target is numerical

if y.dtype != "object":

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    print("Regression MSE:")
    print(mean_squared_error(y_test, pred))
