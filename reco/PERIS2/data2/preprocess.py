import pandas as pd
import sys

file = sys.argv[1]

df = pd.read_csv(file)
df.drop(['cid','act'], axis = 1)
df['ratings'] = 5
df = df[['uid', 'iid', 'ratings', 'time']]
print(len(df))
df.to_csv("Userbehavior.csv", header = False, index = False)



