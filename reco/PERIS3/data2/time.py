import pandas as pd
import sys
file = sys.argv[1]
df = pd.read_csv(file)
time = df.iloc[:,3]
print(len(time))
print(min(time))#old
print(max(time))#latest

