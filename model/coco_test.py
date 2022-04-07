import json

dir = r"C:\Users\zhiyuan\Downloads\annotations_trainval2017\annotations\instances_val2017.json"
json = json.load(open(dir, "r"))
print(json["info"])
