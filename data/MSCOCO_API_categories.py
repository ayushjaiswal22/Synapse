#!/usr/bin/env python

"""
This script helps to get the category information for specific MSCOCO classes.
"""

from pycocotools.coco import COCO

rootPath = 'coco'
annPath = 'annotations'
dataType = 'train2014'
annFile = '%s/%s/instances_%s.json' % (rootPath, annPath, dataType)


def print_categories_from_id(category_ids=[]):
    """prints the category representations for the given category IDs. Prints all categories if the parameter is an
     empty list"""

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # if list is empty load all categories
    if len(category_ids) == 0:
        categories = coco.loadCats(coco.getCatIds())
    else:
        categories = coco.loadCats(category_ids)

    print "\ncategories : "

    for item in categories:
        print str(item)


def print_categories_from_name(cat_strings=[]):
    """Prints the category representations for the given category names. Prints all categories if the parameter is an
     empty list"""

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # if list is empty load all categories
    if len(cat_strings) == 0:
        categories = coco.loadCats(coco.getCatIds())
    else:
        category_ids = coco.getCatIds(catNms=cat_strings)
        categories = coco.loadCats(category_ids)

    print "\ncategories : "

    for item in categories:
        print str(item)


if __name__ == '__main__':

    # Example calls
    print_categories_from_id([4, 10])
    # print_categories_from_name(["person", "dog", "cake"])
