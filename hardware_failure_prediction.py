from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

df = pd.read_csv('/content/drive/MyDrive/predictive_maintenance_dataset.csv')

total_devices = len(df.device.unique())
print('Tem um total de {} dispositivos'.format(total_devices))

total_failure_devices = len(df[df.failure == 1].device.unique())
print('Tem um total de {} dispositivos que falharam'.format(total_failure_devices))

df.date = pd.to_datetime(df.date)
df['activedays'] = df.date - df.date[0]
df['month'] = df['date'].dt.month
df['week_day'] = df.date.dt.weekday
df['week_day'].replace(0, 7, inplace=True)

df_date = df.groupby('device').agg({'date': max})
df_failure = df.loc[df.failure == 1, ['device', 'date']]
df_good = df.loc[df.failure == 0, ['device', 'date']]

df['max_date'] = df.device.map(df_date.date.to_dict())
dff = df[(df.failure == 1) & (df.date != df.max_date)]

df1 = df.groupby('device').agg({'date': max}).reset_index()
df = df.reset_index(drop=True)
df2 = pd.merge(df1, df, how='left', on=['device', 'date'])

df2['failure_before'] = 0
failure_devices = ['S1F136J0', 'W1F0KCP2', 'W1F0M35B', 'S1F0GPFZ', 'W1F11ZG9']
df2.loc[df2.device.isin(failure_devices), 'failure_before'] = 1

df2.device = df2.device.str[:4]

cat_ftrs = ['metric3', 'metric4', 'metric5', 'metric7', 'metric9']
for col in cat_ftrs:
    df2[col] = df2[col].astype('object')

df2.activedays = df2.activedays.astype('str').apply(lambda x: x.split(' ')[0]).astype('int')

for col in ['month', 'week_day']:
    df2[col] = df2[col].astype('object')

df2['metric1'] = np.log(1 + df2['metric1'])

scaler = StandardScaler()
num_ftrs = ['metric1', 'metric2', 'metric6']
df2[num_ftrs] = scaler.fit_transform(df2[num_ftrs])

df.drop('metric8', axis=1, inplace=True)
df2.drop(['date', 'max_date'], axis=1, inplace=True)

df2 = pd.get_dummies(df2, drop_first=True)

X = df2.drop('failure', axis=1)
Y = df2.failure

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf.fit(X, Y)

features = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_}).sort_values(by='importance', ascending=False).set_index('feature')
model = SelectFromModel(clf, prefit=True)
x_reduced = model.transform(X)
x_reduced = pd.DataFrame(x_reduced)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    print(f"Acurácia do {model.__class__.__name__}: {accuracy_score(Y_test, Y_pred)}")
    print(f"Precisão: {precision_score(Y_test, Y_pred)}")
    print(f"Recall: {recall_score(Y_test, Y_pred)}")
    print(f"F1 Score: {f1_score(Y_test, Y_pred)}")
    print(classification_report(Y_test, Y_pred))

    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model.__class__.__name__} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    if hasattr(model, "predict_proba"):
        Y_prob = model.predict_proba(X_test)[:, 1]
    else:
        Y_prob = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model.__class__.__name__} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, Y_train)
evaluate_model(svm_clf, X_test, Y_test)

rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf.fit(X_train, Y_train)
evaluate_model(rf_clf, X_test, Y_test)

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, Y_train)
evaluate_model(gb_clf, X_test, Y_test)

ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_clf.fit(X_train, Y_train)
evaluate_model(ada_clf, X_test, Y_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
evaluate_model(knn, X_test, Y_test)

xgb_clf = xgb.XGBClassifier(random_state=42)
xgb_clf.fit(X_train, Y_train)
evaluate_model(xgb_clf, X_test, Y_test)