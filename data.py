import os

# Idea: getting rid of the empty lines in the dataset? 

# Using the lovecraft stories dataset
dataset_path = 'dataset'
text = ''

for file in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, file)
    print(f'Reading {file_path}')
    with open(file_path, 'r', encoding='utf-8') as f:
        text += f.read() + '\n'
        print('Done')

# Combining and saving the stories in a single file
dataset_combined = 'lovecraft-stories.txt'
with open(dataset_combined, 'w', encoding='utf-8') as f:
    f.write(text)
