from pycocotools.coco import COCO
import json


def instance_count(annotation_path):
    result = []
    coco = COCO(annotation_file=annotation_path)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for name in nms:
        cat_id = coco.getCatIds(catNms=name)
        img_ids = coco.getImgIds(catIds=cat_id)

        j = json.load(open(annotation_path, "r"))
        count = 0
        for ann in j['annotations']:
            if ann['category_id'] == cat_id[0]:
                count += 1
        # print("class {} has {} instance in {} images".format(name, count, len(img_ids)))
        result.append({'cls': name, 'count': count, 'img_ids': img_ids})
    return result


if __name__ == "__main__":
    annotation_file = r"D:\temp\coco\annotations.json"
    a = instance_count(annotation_file)
    print(a)
