import pandas as pd

df = pd.read_csv(r"D:\deepfake-detector\faceforensic++\csv\all_data.csv")
print(df["Label"].value_counts())   # Fake vs Real
print(df.groupby("Label")["File Path"].count())   # same thing
