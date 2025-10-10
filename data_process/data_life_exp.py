import pandas as pd
cities = ['Birmingham', 'Leeds', 'Liverpool']

input_csv_paths = [ "",]
for input_csv_path in input_csv_paths:
    df = pd.read_csv(input_csv_path)
    base_name = input_csv_path.split('/')[-1].split('.')[0] 
    for city in cities:
        city_df = df[df['CityName'] == city]  
        output_csv_path = f"{city}_{base_name}.csv"   
        city_df.to_csv(output_csv_path, index=False)
