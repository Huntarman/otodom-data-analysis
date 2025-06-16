import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression

df = pd.read_csv("otodom_all.csv", sep=';', engine='python')

df = df.dropna(subset=['cenaPLN', 'powierzchnia_mkw'])
df['cena_za_m2'] = df['cenaPLN'] / df['powierzchnia_mkw']

features = [
    'zmywarka', 'lodowka', 'meble', 'piekarnik', 'kuchenka',
    'pralka', 'telewizor', 'telewizja_kablowa', 'internet', 'telefon'
]

freq = {}
for feature in features:
    freq[feature] = df[feature].sum()


pairs = list(itertools.combinations(features, 2))

pair_freq = {}
for f1, f2 in pairs:
    colname = f"{f1}__{f2}"
    df[colname] = df[f1] * df[f2]
    pair_freq[colname] = df[colname].sum()


X = df[features]
y = df['cenaPLN']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'udogodnienie': X.columns,
    'wplyw_na_cene': model.coef_,
    'liczba_wystapien': [freq[col] for col in X.columns]
}).sort_values(by='wplyw_na_cene', ascending=False)

print("Wpływ udogodnień na cenę:")
print(results)

X = df[features]
y = df['cena_za_m2']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'udogodnienie': X.columns,
    'wplyw_na_cene_m2': model.coef_,
    'liczba_wystapien': [freq[col] for col in X.columns]
}).sort_values(by='wplyw_na_cene_m2', ascending=False)

print("Wpływ udogodnień na cenę za m²:")
print(results)

X = df[[f"{f1}__{f2}" for f1, f2 in pairs]]
y = df['cenaPLN']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'para': X.columns,
    'wplyw_na_cene': model.coef_,
    'liczba_wystapien': [pair_freq[col] for col in X.columns]
}).sort_values(by='wplyw_na_cene', ascending=False)

print("Wpływ kombinacji udogodnień na cenę):")
print(results)

X = df[[f"{f1}__{f2}" for f1, f2 in pairs]]
y = df['cena_za_m2']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'para': X.columns,
    'wplyw_na_cene_m2': model.coef_,
    'liczba_wystapien': [pair_freq[col] for col in X.columns]
}).sort_values(by='wplyw_na_cene_m2', ascending=False)

print("Wpływ kombinacji udogodnień na cenę za m²:")
print(results)