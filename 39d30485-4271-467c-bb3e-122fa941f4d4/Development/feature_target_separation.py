#Separate Features & Target
X = df.drop(columns=['id', 'target'])
y = df['target']
#Checking missing percentage
missing_percent = X.isnull().mean() * 100
missing_percent.sort_values(ascending=False).head(10)
missing_cols = missing_percent[missing_percent > 0].index
print(len(missing_cols))
missing_percent.sort_values(ascending=False)
print(missing_percent.head(20))
