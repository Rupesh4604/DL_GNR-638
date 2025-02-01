from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from xml.sax.handler import all_features

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../data/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agr', 'pln', 'bbd', 'bch', 'bld', 'chp', 'drs',
                   'for', 'frw', 'gof', 'hrb', 'int', 'mrs', 'mhp',
                   'ops', 'pkb', 'riv', 'rwy', 'srs', 'stg', 'tns']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 69. For
#simplicity, we assume this is the number of test cases per category, as
#well.

# NUM_TRAIN_PER_CAT = 70


def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES)
        
    print("Train labels:",np.shape(train_labels))
    print("Test labels:", np.shape(test_labels))

    # train_image_paths, test_image_paths, train_labels, test_labels = \
    #     get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # YOU CODE get_tiny_images.py 
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)
        val_image_feats = get_tiny_images(val_image_paths)

    elif FEATURE == 'bag_of_sift':
        # YOU CODE build_vocabulary.py
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 200  ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats_1.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            # print('train_image_feats: ', np.shape(train_image_feats))
            with open('train_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_1.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_1.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            # print("test features:",np.shape(test_image_feats))
            with open('test_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_1.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
        
        if os.path.isfile('val_image_feats_1.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths);
            with open('val_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_1.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)
    elif FEATURE == 'dumy_feature':
        train_image_feats = []
        test_image_feats = []
        val_image_feats = []
    else:
        raise NameError('Unknown feature type')

    # TODO Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    if CLASSIFIER == 'nearest_neighbor':
        # YOU CODE nearest_neighbor_classify.py
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER == 'support_vector_machine':
        # YOU CODE svm_classify.py
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER == 'dumy_classifier':
        # The dummy classifier simply predicts a random category for
        # every test case
        predicted_categories = test_labels[:]
        shuffle(predicted_categories)
    else:
        raise NameError('Unknown classifier type')
    
    print("Train features shape:", train_image_feats.shape)
    print("Test features shape:", test_image_feats.shape)

    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    
    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section. 
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    #visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)
    print("Concatenating features and labels for t-SNE visualization")
    all_feats, all_labels = combine_features_and_labels(train_image_feats, test_image_feats, val_image_feats, train_labels, test_labels, val_labels)
    print("Visualizing t-SNE")
    visualize_tsne(all_feats, all_labels, CATEGORIES,save_path="tsne_visualization.png")

def combine_features_and_labels(train_feats, test_feats, val_feats, train_labels, test_labels, val_labels):
    """
    Combine features and labels from training, testing, and validation datasets.
    """
    # Combine all features and labels into single arrays
    all_feats = np.vstack([train_feats, test_feats, val_feats])
    all_labels = train_labels + test_labels + val_labels  # Concatenate label lists
    return all_feats, all_labels

def visualize_tsne(features, labels, categories, perplexity=30, learning_rate=200, save_path="tsne_visualization.png"):
    """
    Visualize high-dimensional features using t-SNE and save the plot.
    
    Args:
        features (ndarray): High-dimensional feature vectors.
        labels (list): Labels corresponding to features.
        categories (list): List of category names.
        perplexity (int): Perplexity parameter for t-SNE.
        learning_rate (int): Learning rate for t-SNE.
        save_path (str): Path to save the t-SNE visualization image.
    """
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Map labels to numeric values for coloring
    label_to_idx = {label: idx for idx, label in enumerate(categories)}
    numeric_labels = [label_to_idx[label] for label in labels]

    # Scatter plot of t-SNE reduced features
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=numeric_labels,
        cmap="tab10",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(scatter, ticks=range(len(categories)), label="Categories")
    plt.title("t-SNE Visualization of SIFT Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    # Manually create category legend
    for label, color in zip(categories, plt.cm.tab10.colors):
        plt.scatter([], [], c=[color], label=label, s=50, edgecolors="k")
    plt.legend(title="Categories", loc="best")

    # Save the plot to a file
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("t-SNE visualization saved to: ", save_path)

    plt.show()



def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')
    plt.savefig("confusion_matrix")
    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
