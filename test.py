import cv2 as cv

list1 = [1, 2, 3]
list2 = [2, 3, 4]


def t(l1, l2):
    l1[1] = 0
    l2[2] = 0
    return l1, l2


l1 , l2 = t(list1, list2)
print(l1, l2)