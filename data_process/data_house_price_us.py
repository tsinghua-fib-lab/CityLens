import pandas as pd
import json
csv_path = ''
txt_path = ''
city = 'Chicago' 
ct_city = 0
df_csv = pd.read_csv(csv_path)
df_txt = pd.read_csv(txt_path, delimiter='|')
def fix_id(x):
    if pd.isnull(x):
        return None
    x = str(x)
    if x.endswith('.0'):
        x = x[:-2]
    if len(x) < 11:
        x = x.zfill(11) 
    return x

ct_to_img = {}
df_csv['RegionName'] = df_csv['RegionName'].map(fix_id)
df_txt['GEOID_ZCTA5_20'] = df_txt['GEOID_ZCTA5_20'].map(fix_id)
df_txt['GEOID_TRACT_20'] = df_txt['GEOID_TRACT_20'].map(fix_id)
df_city = df_csv[df_csv['City'] == city][['RegionName', 'City', '2020-06-30']].copy()
ct_to_zcta = {}
for _, row in df_txt.iterrows():
    ct = row['GEOID_TRACT_20']
    zcta = row['GEOID_ZCTA5_20']
    if ct not in ct_to_zcta:
        ct_to_zcta[ct] = set()
    ct_to_zcta[ct].add(zcta)
results = []
for ct, region_names in ct_to_zcta.items():
    region_names = list(region_names)
    sub_df = df_city[df_city['RegionName'].isin(region_names)]
    if len(sub_df) == 0:
        continue 
    if len(sub_df) < len(region_names):  
        ct_city += 1
    price_mean = sub_df[''].astype(float).mean()
    region_name_list = sub_df[''].tolist()
    city_name = sub_df['City'].iloc[0]
    results.append({
        'ct': ct,
        'region_names': ';'.join(region_name_list), 
        'city': city_name,
        '': price_mean,
    })
df_result = pd.DataFrame(results)
city_stan = city.replace(" ", "")
df_result.to_csv(f'{city_stan}_ct_house_price_avg.csv', index=False)
