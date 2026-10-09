import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------------------------
# Load the 90-dimensional AAL feature vectors
# -------------------------------------------------

train = pd.read_csv(
    "Output/train/AAL_statistics_volumn_train.csv",
    header=None
)

test = pd.read_csv(
    "Output/test/AAL_statistics_volumn_test.csv",
    header=None
)


# First column = subject ID
train_ids = train.iloc[:, 0].astype(str).str.strip()
test_ids = test.iloc[:, 0].astype(str).str.strip()

# Remaining 90 columns = AAL grey-matter volumes
X_train = train.iloc[:, 1:].astype(float)
X_test = test.iloc[:, 1:].astype(float)


# -------------------------------------------------
# Load training labels
# -------------------------------------------------

labels = pd.read_csv(
    "Original/data_labels.csv",
    skipinitialspace=True
)

labels.columns = labels.columns.str.strip()
labels["File Name"] = labels["File Name"].astype(str).str.strip()

# Convert Data_00.nii -> Data_00
labels["Subject"] = labels["File Name"].str.replace(
    ".nii", "",
    regex=False
)

label_map = dict(zip(labels["Subject"], labels["Label"]))

y_train = train_ids.map(label_map)


# Make sure every training subject has a label
if y_train.isna().any():
    missing = train_ids[y_train.isna()].tolist()
    raise ValueError(f"Missing labels for: {missing}")


print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)

print("\nClass distribution:")
print(y_train.value_counts())


# -------------------------------------------------
# SVM
#
# StandardScaler performs feature normalisation.
# Scaling is fitted separately inside each CV fold,
# preventing information leakage.
# -------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="linear",
        C=1.0
    )
)


# -------------------------------------------------
# 5-fold cross-validation on training set
# -------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv
)

cv_predictions = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv
)

print("\n==============================")
print("5-FOLD CROSS-VALIDATION")
print("==============================")

print("Fold accuracies:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(
    y_train,
    cv_predictions,
    labels=["AD", "NC"]
))

print("\nClassification report:")
print(classification_report(
    y_train,
    cv_predictions,
    labels=["AD", "NC"]
))


# -------------------------------------------------
# Train final model using ALL 40 training subjects
# -------------------------------------------------

model.fit(X_train, y_train)


# -------------------------------------------------
# Predict the 10 unseen test subjects
# -------------------------------------------------

test_predictions = model.predict(X_test)

results = pd.DataFrame({
    "File Name": test_ids + ".nii",
    "Prediction": test_predictions
})

results.to_csv(
    "Output/test/SVM_predictions.csv",
    index=False
)


print("\n==============================")
print("TEST SET PREDICTIONS")
print("==============================")

print(results.to_string(index=False))

print(
    "\nPredictions saved to "
    "Output/test/SVM_predictions.csv"
)