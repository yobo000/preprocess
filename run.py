# -*- coding: utf-8 -*-

import os
from datautil import CategoriesDataset


def main():
    root_dir = os.path.join(os.path.abspath(os.path.dirname(
                            os.path.dirname(__file__))))
    data_dir = os.path.join(root_dir, "data")
    csv_file = os.path.join(root_dir, "file", "test_dev.csv")
    data = CategoriesDataset("images")
    dataset = data(csv_file, data_dir)
    return dataset[0]


if __name__ == '__main__':
    print(main())
