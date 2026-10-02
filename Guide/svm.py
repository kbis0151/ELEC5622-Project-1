# Usage: python svm.py   (run in a folder containing the two CSVs from Measurement and data_labels.csv)
# Needs: pip install pandas scikit-learn openpyxl
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# One row per subject: name, then one GM volume per AAL region. Keep the first 90 ROIs.
tr = pd.read_csv('AAL_statistics_volumn_train.csv', header=None, index_col=0).iloc[:, :90]
te = pd.read_csv('AAL_statistics_volumn_test.csv', header=None, index_col=0).iloc[:, :90]
tr.columns = te.columns = [f'ROI_{i}' for i in range(1, 91)]

lab = pd.read_csv('data_labels.csv', skipinitialspace=True)
lab = lab.set_index(lab['File Name'].str.replace('.nii', '', regex=False))['Label']
y = lab.loc[tr.index]

# Scaler inside the pipeline -> fitted on training folds only (no leakage).
clf = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                   {'svc__kernel': ['linear', 'rbf'], 'svc__C': [0.01, 0.1, 1, 10, 100]},
                   cv=LeaveOneOut())
clf.fit(tr, y)
print('Best parameters:', clf.best_params_)
print(f'Leave-one-out accuracy on training set: {clf.best_score_:.3f}')

pred = pd.Series(clf.predict(te), index=(te.index + '.nii').rename('File Name'), name='Prediction')
print(pred.to_string())
pred.to_csv('predictions.csv')

# Submission spreadsheet: 90 ROI GM volumes for all 50 subjects.
pd.concat([tr, te]).sort_index().to_excel('ROI_GM_volumes_all_subjects.xlsx', index_label='Subject')
print('Wrote predictions.csv and ROI_GM_volumes_all_subjects.xlsx')
