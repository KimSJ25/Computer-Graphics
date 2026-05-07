import numpy as np
from collections import Counter
import torchvision
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"Data shape - X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    num_folds = 5
    k_choices = [1, 3, 5, 7, 9]
    metrics = ['L1', 'L2']
    
    # Split training data into 5 folds for Cross Validation
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    best_acc = 0
    best_k = 1
    best_metric = 'L1'
    
    accuracies_history = {metric: [] for metric in metrics}
    fold_accuracies_history = {metric: {} for metric in metrics}

    for metric in metrics:
        print(f"\n--- Cross Validation for {metric} Distance ---")
        k_to_accuracies = {k: [] for k in k_choices}

        for i in range(num_folds):
            # Set the current fold as validation set, others as training set
            X_val_fold = X_train_folds[i]
            y_val_fold = y_train_folds[i]
            
            X_tr_fold = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
            y_tr_fold = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
            
            # Calculate distances for the current fold
            dists = compute_distances(X_tr_fold, X_val_fold, metric=metric)
            
            for k in k_choices:
                y_val_pred = predict_labels(dists, y_tr_fold, k=k)
                accuracy = float(np.sum(y_val_pred == y_val_fold)) / len(y_val_fold)
                k_to_accuracies[k].append(accuracy)

        fold_accuracies_history[metric] = k_to_accuracies

        for k in k_choices:
            # Average accuracy across all 5 folds
            avg_acc = np.mean(k_to_accuracies[k])
            print(f"Metric: {metric}, K: {k} => CV Avg Accuracy: {avg_acc:.4f}")
            accuracies_history[metric].append(avg_acc)
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_k = k
                best_metric = metric
        print("-" * 30)

    # Evaluate the best model on the actual test set
    print(f"\n[Best Model] Metric: {best_metric}, K: {best_k} (CV Accuracy: {best_acc:.4f})")
    print("Evaluating best model on the Test Dataset...")
    best_dists = compute_distances(X_train, X_test, metric=best_metric)
    best_pred = predict_labels(best_dists, y_train, k=best_k)
    test_acc = float(np.sum(best_pred == y_test)) / len(y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    print("\n[Confusion Matrix of the best model]")
    cm = confusion_matrix(y_test, best_pred)
    print(cm)

    print("\n[Plotting the results]")
    plt.figure(figsize=(10, 6))
    
    colors = {'L1': 'blue', 'L2': 'orange'}
    
    for metric in metrics:
        k_points = []
        acc_points = []
        for k in k_choices:
            for acc in fold_accuracies_history[metric][k]:
                k_points.append(k)
                acc_points.append(acc)
        plt.scatter(k_points, acc_points, color=colors[metric], alpha=0.4, label=f'{metric} Individual Folds')
        
        plt.plot(k_choices, accuracies_history[metric], marker='o', color=colors[metric], linewidth=2, label=f'{metric} Average')
        
    plt.title('KNN Cross-Validation Accuracy on CIFAR-10')
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.xticks(k_choices)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()