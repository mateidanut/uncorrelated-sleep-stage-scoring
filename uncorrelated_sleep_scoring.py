#!/usr/bin/env python
# coding: utf-8
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.io import loadmat
import multiprocessing

rng = default_rng(42)


def load_data_spectrogram(eeg_files, idx):
    # load data from files stoared as spectrograms
    with h5py.File(eeg_files[idx], 'r') as f:
        X = np.array(f['X'])
        labels = np.array(f['label'])

    X = np.moveaxis(X, -1, 0)
    X = np.moveaxis(X, -1, 1)

    # average the spectrogram on the time axis
    X_avg = X.mean(axis=1)

    # read the labels from the file, but make them 0 indexed
    y_true = (labels.T - 1).ravel()
    
    # get rid of missing values
    valid = ~(np.isinf(X_avg) | np.isnan(X_avg))
    valid = np.all(valid, axis=1)
    X_avg, y_true = X_avg[valid], y_true[valid]
    
    return X_avg, y_true


def load_data_signal(scans, idx):
    # load data from files stored as signals
    scan_mat = loadmat(scans[idx])
    data, _, labels = scan_mat['data'], scan_mat['epoch'], scan_mat['labels']

    return data[..., 0], labels


class DataSampler():
    # class used for data sampling
    def __init__(self, data_type='spectrogram'):
        if data_type == 'signal':
            data_files = ['./raw_data/' + f for f in sorted(os.listdir('./raw_data'))]
            load_data = load_data_signal
        elif data_type == 'spectrogram':
            data_files = ['./mat/' + f for f in sorted(os.listdir('./mat')) if f.endswith('eeg.mat')]
            load_data = load_data_spectrogram
        else:
            raise ValueError('data_type can only be "spectrogram" or "signal"!')

        # load the whole data in memory from the start for faster sampling
        all_data = [load_data(data_files, idx) for idx in range(len(data_files))]
        self.X_full, self.y_full = zip(*all_data)
        self.data_count = len(data_files)


    def __call__(self, indices, n_samples_per_class=None, classes=None):
        # sample data of specific patients
        X_avg = []
        y_true = []

        for idx in indices:
            X, y = self.X_full[idx], self.y_full[idx]

            if n_samples_per_class is None:
                X_avg.append(X)
                y_true.append(y)
                continue

            p = rng.permutation(np.arange(len(y)))
            # shuffle X and y
            X, y = X[p, :], y[p]

            # sample equally from each class, for each patient
            if classes is None:
                classes = list(range(5))

            for cls in classes:
                sample = np.where(y == cls)[0][:n_samples_per_class]
                X_avg.append(X[sample])
                y_true.append(y[sample])
                
        return np.concatenate(tuple(X_avg)), np.concatenate(tuple(y_true))



def train_pcas(sample_patient_data, X_train, n_samples_per_class):
    pcas = []

    for cls in range(5):
        X, _ = sample_patient_data(X_train, n_samples_per_class=n_samples_per_class, classes=[cls])

        model = PCA(n_components=7)
        model.fit(X)
        pcas.append(model)
        
    return pcas


def sample_classifier_data(sample_patient_data, patients, n_samples_per_class):
    X_train, y_train = sample_patient_data(patients, n_samples_per_class=n_samples_per_class)
    
    return X_train, y_train


def train_classifiers(pcas, X_train, y_train):
    # preprocess with pca
    X_train_sparse = pcas[0].transform(X_train)

    for pca in pcas[1:]:
        X_pca = pca.transform(X_train)
        X_train_sparse = np.hstack((X_train_sparse, X_pca))

    
    # define models
    knn = KNeighborsClassifier(7)
    svm = SVC(kernel="linear", C=0.025)
    mlp = MLPClassifier(alpha=1, max_iter=1000, activation='relu')

    dummy = DummyClassifier(strategy='constant', constant=2)
    one_nn = KNeighborsClassifier(1)

    classifiers = [knn, svm, mlp, dummy, one_nn]

    # standardize data first
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_sparse)

    for model in classifiers:
        model.fit(X_train_std, y_train)
        
    return scaler, classifiers


def fit_models(sample_patient_data, train_patients, n_samples_per_class, *, pca_classif_thresh=60):
    patients_pca = train_patients[:pca_classif_thresh]
    patients_classif = train_patients[pca_classif_thresh:]
    
    pcas = train_pcas(sample_patient_data, patients_pca, n_samples_per_class)
    X_train, y_train = sample_classifier_data(sample_patient_data, patients_classif, n_samples_per_class)
    
    scaler, classifiers = train_classifiers(pcas, X_train, y_train)
    return pcas, scaler, classifiers


def test_models(X_test, y_test, pcas, scaler, classifiers):
    X_pcas = pcas[0].transform(X_test)

    for pca in pcas[1:]:
        X_pca = pca.transform(X_test)
        X_pcas = np.hstack((X_pcas, X_pca))

    X_test_std = scaler.transform(X_pcas)
    
    results = []
    for model in classifiers:
        y_pred = model.predict(X_test_std)
        
        d = classification_report(y_test, y_pred, output_dict=True)
        results.append((d['accuracy'], d['macro avg']['f1-score']))
        
    return list(zip(*results))


def plot_results(n_samples, results):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Classification perfomance on the test set')
    
    for i, (metric, metric_name) in enumerate([('acc', 'Accuracy'), ('f1', 'F1-score')]):
        test_res = list(zip(*results[metric]))
        
        axs[i].plot(n_samples, test_res[0], 'r-', label='knn')
        axs[i].plot(n_samples, test_res[1], 'g-', label='svm')
        axs[i].plot(n_samples, test_res[2], 'b-', label='mlp')

        if metric == 'acc':
            # for the accuracy we also plot the "dummy" and Bayes Optimal classifiers
            axs[i].plot(n_samples, test_res[3], 'o-', label='majority_class')

            one_nn_error = 1 - np.array(test_res[4])
            bayes_error = one_nn_error / 2
            bayes_accuracy = 1 - bayes_error
            axs[i].plot(n_samples, bayes_accuracy, 'o-', label='bayes_classifier')

        axs[i].set(xlabel='Number of samples / class / patient', ylabel=metric_name)
        axs[i].legend()
        axs[i].grid()
    
    plt.show()


def run_experiment():
    # function to call for training and testing the models,
    # followed by plotting of results
    
    eeg_sampler = DataSampler()
    data_count = eeg_sampler.data_count

    TRAIN_SIZE = int(.8 * data_count)
    patient_permutation = rng.permutation(data_count)

    # split the patients into train and test sets
    train_patients = patient_permutation[:TRAIN_SIZE]
    test_patients = patient_permutation[TRAIN_SIZE:]
    X_test, y_test = eeg_sampler(test_patients)

    print(len(train_patients), len(test_patients))
    
    results = {
        'acc': [],
        'f1': [],
    }
    
    n_samples = range(1, 90, 1)

    # start a pool of processes to run the experiment in parallel
    pool = multiprocessing.Pool(20)
    args = [(eeg_sampler, train_patients, X_test, y_test, i) for i in n_samples]
    output = pool.starmap(train_test_classifiers_global, args)
    results['acc'], results['f1'] = zip(*output)
        
    # plot the final accuracies and f1 scores for the trained models
    plot_results(n_samples, results)


def train_test_classifiers_global(eeg_sampler, train_patients, X_test, y_test, train_samples_per_class):
    acc_tests, f1_tests = [], []
    # use boosting to make results more consistent
    for _ in range(10):
        pcas, scaler, classifiers = fit_models(eeg_sampler, train_patients, train_samples_per_class)
        acc_test, f1_test = test_models(X_test, y_test, pcas, scaler, classifiers)
        acc_tests.append(acc_test)
        f1_tests.append(f1_test)

    mean_acc = tuple(np.array(acc_tests).mean(axis=0))
    mean_f1 = tuple(np.array(f1_tests).mean(axis=0))

    return mean_acc, mean_f1
