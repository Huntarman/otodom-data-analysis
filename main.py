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
results

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
results

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
results

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
results


dzielnice_encoded = pd.get_dummies(df['dzielnica'])

interaction_cols = []
for feature in features:
    new_cols = []
    for dzielnica in df['dzielnica'].unique():
        col_name = f"{dzielnica}_x_{feature}"
        mask = (df['dzielnica'] == dzielnica)
        df[col_name] = df[feature] * mask.astype(int)
        if df[col_name].sum() > 50:
            interaction_cols.append(col_name)
        else:
            df.drop(columns=[col_name], inplace=True)

X = df[interaction_cols]
y = df['cenaPLN']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'interakcja': X.columns,
    'wplyw_na_cene': model.coef_,
    'liczba_wystapien': [df[col].sum() for col in X.columns]
}).sort_values(by='wplyw_na_cene', ascending=False)

print("Wpływ interakcji udogodnień i dzielnic na cenę:")
print(results)

X = df[interaction_cols]
y = df['cena_za_m2']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'interakcja': X.columns,
    'wplyw_na_cene': model.coef_,
    'liczba_wystapien': [df[col].sum() for col in X.columns]
}).sort_values(by='wplyw_na_cene', ascending=False)

print("Wpływ interakcji udogodnień i dzielnic na cenę za m2:")
print(results)