import random
import json

def main():
    train_size = .70
    test_size = .15
    valid_size = .15

    # Load and combine all csv files
    with open("recipes.json") as recipe_file:
        recipes = json.load(recipe_file)

    random.shuffle(recipes)

    n_items = len(recipes)

    start = 0
    train_end = int(n_items * train_size)
    test_end = int(train_end +  n_items * test_size)
    valid_end = n_items - 1
    
    # write a train, test and valid set
    with open("train.json", "wb") as f:
        json.dump(recipes[0: train_end], f)

    with open("test.json", "wb") as f:
        json.dump(recipes[train_end: test_end], f)

    with open("valid.json", "wb") as f:
        json.dump(recipes[test_end: valid_end], f)


if __name__ == '__main__':
    main()