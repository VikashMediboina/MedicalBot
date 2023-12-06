import csv
import json

# Read JSON data from the file
with open('HealthCareMagic-100k (1).json', 'r') as json_file:
    json_data = json.load(json_file)

# with open('HealthCareMagic-100k-updated.json', 'w') as json_file:
#     json_data = json.load(json_file)
# with open('HealthCareMagic-100k-updated.json', 'w') as json_file:
#     json.dump(json_data, json_file, indent=4)
# # Writing data to CSV file with UTF-8 encoding
with open('HealthCareMagic-100k.csv', 'w') as csvfile:
    fieldnames = ['Text', 'Instruction', 'Input', 'Output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Writing CSV header
    writer.writeheader()

    # Writing JSON data to CSV
    for entry in json_data:
        prompt = "below is an instruction that describes a task paired with an input that provides further context Write a response that appropriately completes the request"
        try:
            text = "{}### Instruction:\n{}### Input:\n{}### Output:\n{}".format(
    prompt.encode('ascii', 'ignore').decode('ascii'),
    entry['instruction'].encode('ascii', 'ignore'),
    entry['input'].encode('ascii', 'ignore'),
    entry['output'].encode('ascii', 'ignore')
)
            row_data = {
                'Text': text,
                'Instruction': entry['instruction'],
                'Input': entry['input'],
                'Output': entry['output']
            }
            writer.writerow(row_data)
        except Exception as e:
            print(e)

print("CSV file has been created successfully.")
