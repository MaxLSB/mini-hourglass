def main():
    # Making and saving the vocabulary for the model
    with open('lovecraft-stories.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # get a list of all unique characters in the dataset
    chars = sorted(list(set(text)))

    # save the vocabulary to a file
    vocab_file = 'model_assets/vocab.txt'

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for char in chars:
            f.write(char + '\n')


if __name__ == "__main__":
    main()
