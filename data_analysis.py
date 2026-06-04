import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv('data/industrial_robot_control_6G_network.csv')

#Klassifiserer datasettet, ser hvordan datasetett ser ut før en setter opp modell

print(df.head(10)) #printer 5 første rader av alle kolonner
print(df.columns) #printer navne på kolonnene
print("Size of dataset")
print(df.shape) #printer str på matrisa, antall rader og kolonner
print("Antall rader per kolonne med manglende data:")
print(df.isnull().sum()) #printer hvor mange data som mangler
# After this initial analysis
# And there is no empty rows


print(df["task_type"].value_counts())
#The data is very evenly constributed across the three task types
print(df["robot_id"].value_counts())
# robot_id wont contribute to the lda or mlp because there isnt information linkin to task type
print(df["timestamp"].value_counts())
print(pd.crosstab(df["timestamp"],df["task_type"]))
#Timestamp doesent seem to impose any useful information, 2025-01-21 and 10 is constant, and the rest looks random, but will extract minutes and put it in the model
print(pd.crosstab(df["position_coordinates"],df["task_type"]))
#position_coordinates was also removed because each observation contains a unique coordinate set. The variable does not represent a reusable pattern and behaves more like an identifier than a predictive feature.
print(df["sensor_id"].value_counts())
print(pd.crosstab(df["sensor_id"], df["task_type"]))
#sensor id deasent seem to impose data leakage

# From the initial analysis:
# - robot_id can be removed
# - position_coordinates can be removed
# - timestamp should be converted to a usable format or removed
# - sensor_type, network_type, sensor_id and slice_id require get_dummies()