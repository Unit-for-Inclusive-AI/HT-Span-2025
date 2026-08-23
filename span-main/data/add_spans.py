# Add Spans (locations in the text)
import pandas as pd

df = pd.read_csv("marathi_combined_data.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.rename(columns={'label': 'class'})

df['class'] = df['class'].replace(['NA', None], 'None')

def expand_positions(position, class_value):
  
    if class_value == 'None':
        return []
    
    try:
        if pd.isna(position) or position.strip() == "":
            return None
        a, b = map(int, position.strip('[]').split(','))
        return list(range(a, b + 1))
    except (ValueError, TypeError):
        return None

df['label'] = df.apply(lambda row: expand_positions(row['positions'], row['class']), axis=1)

df.to_csv("marathi_combined_updated_data.csv", index=False)

print("Data processing completed and saved to 'output_file.csv'.")