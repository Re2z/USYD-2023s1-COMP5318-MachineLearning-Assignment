# Import all libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset = pd.read_csv('../test-before.csv')

# Pre-process the dataset

# Replace the classes 'class1' 'class2' to 0 1 and replace the missing value ? to NA
dataset.replace(['?', 'class1', 'class2'], [np.nan, 0, 1], inplace=True)
print(dataset)
# Separate the dataset to the features and label

# Using the sklearn.impute.SimpleImputer to replace the missing value to the mean value of the column
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_data = imp_mean.fit_transform(dataset)

print(type(X_data))