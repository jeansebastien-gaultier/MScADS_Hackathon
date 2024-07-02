# File used to transform the rows in the csv file into individual markdown files.

import pandas as pd
import os

# Load the dataset
filepath = "/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/mtsamples_with_rand_names.csv"
df = pd.read_csv(filepath)

# Create a directory to store the Markdown files
output_dir = "/Users/jean-sebastiengaultier/Desktop/UChicago/Academic/Hackathon/data"
os.makedirs(output_dir, exist_ok=True)

# Function to create Markdown file for each row
def row_to_markdown(row, output_dir):
    file_name = f"{row['Unnamed: 0']}.md"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"Patient Data: {row['Unnamed: 0']} \n\n ")
        file.write(f"Name: {row['first_name']} {row['last_name']} \n\n ")
        file.write(f"Sample Name: {row['sample_name']} \n\n ")
        file.write(f"Medical Field: {row['medical_specialty']} \n\n ")
        file.write(f"Description: {row['description']} \n\n ")
        file.write(f"Keywords: {row['keywords']} \n\n ")
        file.write(f"Transcription: {row['transcription']} \n\n ")

# Iterate over each row and create Markdown file
for index, row in df.iterrows():
    row_to_markdown(row, output_dir)