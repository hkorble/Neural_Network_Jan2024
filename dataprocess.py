import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Read the dataset
df = pd.read_csv('./student-mat.csv', sep=';')


pd.set_option('future.no_silent_downcasting', True)
df.replace({'yes': 1, 'no': 0}, inplace=True)
pd.set_option('display.max_columns', None)


df = pd.get_dummies(df, columns=["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian" ])
df.replace({True: 1, False: 0}, inplace=True)
#print(df.shape) # (395, 51)

X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

