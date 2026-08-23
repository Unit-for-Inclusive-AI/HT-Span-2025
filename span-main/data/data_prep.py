# JSON to CSV
import json
import csv

with open("tamil_4.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)


csv_data = []
for item in data:
    for entity in item.get('entities', []):
        offset_list = list(range(entity['start_offset'], entity['end_offset'] + 1))
        
        csv_row = {
            'id': item['id'],
            'text': item['text'],
            'label': ','.join(map(str, offset_list)),
            'class_name': entity['label']
        }
        csv_data.append(csv_row)

csv_columns = ['id', 'text', 'label', 'class_name']

with open("/Users/devendrakayande/Desktop/SPAN/tamil/tamil_4_output.csv", "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_data)

print("CSV file created successfully.")
