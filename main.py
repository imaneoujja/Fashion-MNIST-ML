import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes

import time

def main(args):
    """
    Train and evaluate the selected network on Fashion-MNIST.

    Arguments:
        args (Namespace): arguments parsed from the command line (see the end of this file).
    """
    ## 1. Load the data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Prepare the data

    # Hold out a third of the shuffled training data as a validation set
    if not args.test:
        indices = np.random.permutation(xtrain.shape[0])
        taille_validation_set = int(xtrain.shape[0] * (1.0 / 3.0))
        i_validation = indices[: taille_validation_set]
        i_train = indices[taille_validation_set:]
        x_val, y_val = xtrain[i_validation], ytrain[i_validation]
        xtrain, ytrain = xtrain[i_train], ytrain[i_train]
        xtest = x_val
        ytest = y_val
        print("Using Validation Set")

    # Normalize data
    means = np.mean(xtrain, axis=0, keepdims=True)
    stds = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    # Optional dimensionality reduction with PCA
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        exvar = pca_obj.find_principal_components(xtrain)
        print(f"The explained variance of the kept dimensions (in percentage) is{exvar:.2f}%")

        # Project train and validation/test data onto the principal components
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

    ## 3. Build the model (reshaping the data for the CNN and the Transformer)
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes)

    if args.nn_type == "cnn":
        # Reshape data
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        model = CNN(1, n_classes)

    elif args.nn_type == "transformer":
        n_classes = get_n_classes(ytrain)
        input_size = xtrain.shape[1]
        size_image_array = int(np.sqrt(input_size))
        chw = (1, size_image_array, size_image_array)
        xtrain = xtrain.reshape(xtrain.shape[0], 1,int(np.sqrt(xtrain.shape[1])),int(np.sqrt(xtrain.shape[1])))
        xtest = xtest.reshape(xtest.shape[0], 1,int(np.sqrt(xtest.shape[1])),int(np.sqrt(xtest.shape[1])))
        model = MyViT(chw,7,2,64,2,n_classes)

    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    t1 = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    t2 = time.time()
    print("\nTraining took", t2 - t1, "seconds\n")

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    # The test set has no labels, so performance is reported on the validation set
    if not args.test:
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    args = parser.parse_args()
    main(args)