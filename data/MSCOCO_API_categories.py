#!/usr/bin/env python
"""
    py-faster-rcnn-ft - Custom Training of Deep Learning Models for Image Classification
    Copyright (C) 2017  DFKI GmbH

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
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
