import json

dir = r"D:\temp\coco\annotations.json"
json = json.load(open(dir, "r"))
print(json["info"])
