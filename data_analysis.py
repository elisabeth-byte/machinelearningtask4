import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv('data/Industrial_fault_detection.csv')

#Klassifiserer datasettet, ser hvordan datasetett ser ut før en setter opp modell

print(df.head()) #printer 5 første rader av alle kolonner
print(df.columns) #printer navne på kolonnene
print("Size of dataset")
print(df.shape) #printer str på matrisa, antall rader og kolonner
print("Antall rader per kolonne med manglende data:")
print(df.isnull().sum()) #printer hvor mange data som mangler
# After this initial analysis, there is no column with useless information, nor column with data leakage
# And there is no empty rows
# Dont need further analysis and can move on to preprocessing
print(df["Fault_Type"].value_counts())
#The data is very skewed, there is a lot more counts of fault 0