import pandas as pd
csv_file_path = '/content/Beginner_Climate_Change_Dataset_20_Features_1200_Rows.csv'
df = pd.read_csv(csv_file_path)
print(f"Successfully loaded the CSV file from: {csv_file_path}")
display(df.head())
