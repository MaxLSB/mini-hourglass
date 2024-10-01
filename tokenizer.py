# Making and saving the vocabulary for the model

with open('lovecraft-stories.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # get a list of all unique characters in the dataset
vocab_size = len(chars)

# save the vocabulary to a file
vocab_file = 'model_assets/vocab.txt'

with open(vocab_file, 'w', encoding='utf-8') as f:
    for char in chars:
        f.write(char + '\n')
