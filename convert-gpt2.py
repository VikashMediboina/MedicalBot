import pandas as pd

# Load your CSV data
csv_file_path = '/Users/vikashmediboina/PycharmProjects/MedicalChatBot/Data/HealthCareMagic-100k.csv'
df = pd.read_csv(csv_file_path)

# Convert CSV data into dialogue format and save it to a text file
output_file_path = 'chatbot_dataset.txt'
with open(output_file_path, 'w') as output_file:
    for index, row in df.iterrows():
        user_input = row['Input']  # Replace 'input_column' with the actual column name containing user inputs
        bot_response = row['Output']  # Replace 'output_column' with the actual column name containing bot responses
        output_file.write(f"User: {user_input}\n")
        output_file.write(f"Bot: {bot_response}\n")
