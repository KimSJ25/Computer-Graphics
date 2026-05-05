import numpy as np
from collections import Counter
import torchvision
from sklearn.metrics import confusion_matrix
import time

def load_cifar10():
    print("1. Downloading and loading CIFAR-10 dataset...")
    # Download data using torchvision (saved in ./data folder)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Extract data subset (Train 5000, Test 1000)
    X_train = trainset.data[:5000]
    y_train = np.array(trainset.targets)[:5000]
    X_test = testset.data[:1000]
    y_test = np.array(testset.targets)[:1000]

    # Flattening (32x32x3 -> 3072) and convert to float32
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    return X_train_flat, y_train, X_test_flat, y_test

def compute_distances(X_train, X_test, metric='L1'):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    print(f"[{metric}] Starting distance calculation ({num_test} test data)...")
    start_time = time.time()
    
    for i in range(num_test):
        if metric == 'L1':
            # L1 Distance (Manhattan)
            dists[i, :] = np.sum(np.abs(X_train - X_test[i, :]), axis=1)
        elif metric == 'L2':
            # L2 Distance (Euclidean)
            dists[i, :] = np.sqrt(np.sum(np.square(X_train - X_test[i, :]), axis=1))
            
    print(f"Distance calculation completed! (Took {time.time() - start_time:.2f} seconds)")
    return dists

def predict_labels(dists, y_train, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=y_train.dtype)

    for i in range(num_test):
        # Sort by shortest distance and extract top k indices
        closest_y = y_train[np.argsort(dists[i, :])[:k]]
        # Majority vote (choose the most frequent label)
        vote = Counter(closest_y)
        y_pred[i] = vote.most_common(1)[0][0]
        
    return y_pred

def main():
    # 1. Load data
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"Data shape - X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    k_choices = [1, 3, 5, 7, 9]
    metrics = ['L1', 'L2']
    
    best_acc = 0
    best_pred = None

    # 2. Evaluate (for each metric and K value)
    for metric in metrics:
        # Calculate distance for a specific metric once and reuse it.
        dists = compute_distances(X_train, X_test, metric=metric)
        
        for k in k_choices:
            y_test_pred = predict_labels(dists, y_train, k=k)
            # Calculate accuracy
            num_correct = np.sum(y_test_pred == y_test)
            accuracy = float(num_correct) / len(y_test)
            print(f"Metric: {metric}, K: {k} => Accuracy: {accuracy:.4f}")
            
            # Save the best prediction result (for confusion matrix)
            if accuracy > best_acc:
                best_acc = accuracy
                best_pred = y_test_pred
        print("-" * 30)

    # 3. Print the Confusion Matrix of the best model
    print("\n[Confusion Matrix of the best model]")
    cm = confusion_matrix(y_test, best_pred)
    print(cm)

if __name__ == '__main__':
    main()