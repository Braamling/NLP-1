import json
import pickle

with open('data/ingred_list.json') as data:
	d = json.load(data)
	

foods = {}
id_count = 0
for list_of_ingredients in d:
	for ingredient in list_of_ingredients['ingredients']:
		if ingredient not in foods.keys():
			foods[ingredient] = id_count
			id_count+=1

pickle.dump(foods, open('list_of_foods.p', 'wb'))