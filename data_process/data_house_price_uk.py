import pandas as pd
input_csv = 'house_price_msoa.csv' 
target_city = 'Birmingham'  
output_csv = f'{target_city}_house_price.csv' 
df = pd.read_csv(input_csv)
filtered_df = df[
    (df['Time'] == '2020-06') &
    (df['Aggregation'] == 'Mean') &
    (df['CityName'] == target_city)
]

filtered_df.to_csv(output_csv, index=False)
