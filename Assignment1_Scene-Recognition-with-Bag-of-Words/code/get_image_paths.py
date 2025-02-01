import os
from glob import glob

from numpy import shape

def get_image_paths(data_path, categories):
    
    num_train_per_cat = 70  # Number of training samples per category
    num_test_per_cat = 20   # Number of test samples per category
    num_val_per_cat = 10    # Number of validation samples per category

    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []
    val_image_paths = []

    train_labels = []
    test_labels = []
    val_labels = []

    for category in categories:

        image_paths = glob(os.path.join(data_path, 'train', category, '*.jpg'))
        # print(image_paths)
        for i in range(num_train_per_cat):  
            train_image_paths.append(image_paths[i])
            #print(train_image_paths)
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, 'test', category, '*.jpg'))
        for i in range(num_test_per_cat):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)
            
        image_paths = glob(os.path.join(data_path, 'val', category, '*.jpg'))
        for i in range(num_val_per_cat):
            val_image_paths.append(image_paths[i])
            val_labels.append(category)
    print("Train labels:",shape(train_labels))
    print("Test labels:", shape(test_labels))
    print("Val Labels:", shape(val_labels))

    return train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels
    # return train_image_paths, test_image_paths, train_labels, test_labels