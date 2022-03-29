#!/usr/bin/env python
# coding: utf-8


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from tqdm import tqdm
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng(42)

from scipy.io import loadmat

from pympler import asizeof

def load_data_spectrogram(eeg_files, idx):
    with h5py.File(eeg_files[idx], 'r') as f:
        X = np.array(f['X'])
#         y = np.array(f['y'])
        labels = np.array(f['label'])

    X = np.moveaxis(X, -1, 0)
    X = np.moveaxis(X, -1, 1)
#     y = np.moveaxis(y, -1, 0)

    X_avg = X.mean(axis=1)
#     y_true = np.where(y == 1)[1]
    # read the labels from the file, but make them 0 indexed
    y_true = (labels.T - 1).ravel()
    
    valid = ~(np.isinf(X_avg) | np.isnan(X_avg))
    valid = np.all(valid, axis=1)
    X_avg, y_true = X_avg[valid], y_true[valid]
    
    return X_avg, y_true


def load_data_signal(scans, idx):
    scan_mat = loadmat(scans[idx])
    data, _, labels = scan_mat['data'], scan_mat['epoch'], scan_mat['labels']

    return data[..., 0], labels


#def make_data_sampler_slow(data_type='spectrogram'):
#    if data_type == 'signal':
#        data_files = ['./raw_data/' + f for f in sorted(os.listdir('./raw_data'))]
#        load_data = load_data_signal
#    elif data_type == 'spectrogram':
#        data_files = ['./mat/' + f for f in sorted(os.listdir('./mat')) if f.endswith('eeg.mat')]
#        load_data = load_data_spectrogram
#    else:
#        raise ValueError('data_type can only be "spectrogram" or "signal"!')
#
#    def sample_patient_data(indices, n_samples_per_class=None, classes=None):
#        X_avg = []
#        y_true = []
#
##         for idx in tqdm(indices):
#        for idx in indices:
#            X, y = load_data(data_files, idx)
#
#            if n_samples_per_class is None:
#                X_avg.append(X)
#                y_true.append(y)
#                continue
#
#            p = rng.permutation(np.arange(len(y)))
#            # shuffle X and y
#            X, y = X[p, :], y[p]
#
#            # sample equally from each class, for each patient
#            if classes is None:
#                classes = list(range(5))
#
#            for cls in classes:
#                sample = np.where(y == cls)[0][:n_samples_per_class]
##                 print('UNIQ', np.unique(y))
##                 print('CLS', cls, sample)
#                X_avg.append(X[sample])
#                y_true.append(y[sample])
#                
##             print(len(X_avg), X_avg[0].shape)
#
#        return np.concatenate(tuple(X_avg)), np.concatenate(tuple(y_true))
#    
#    return sample_patient_data, len(data_files)


def make_data_sampler(data_type='spectrogram'):
    if data_type == 'signal':
        data_files = ['./raw_data/' + f for f in sorted(os.listdir('./raw_data'))]
        load_data = load_data_signal
    elif data_type == 'spectrogram':
        data_files = ['./mat/' + f for f in sorted(os.listdir('./mat')) if f.endswith('eeg.mat')]
        load_data = load_data_spectrogram
    else:
        raise ValueError('data_type can only be "spectrogram" or "signal"!')

    all_data = [load_data(data_files, idx) for idx in range(len(data_files))]
    X_full, y_full = zip(*all_data)

    def sample_patient_data(indices, n_samples_per_class=None, classes=None):
        X_avg = []
        y_true = []

#         for idx in tqdm(indices):
        for idx in indices:
            #X, y = load_data(data_files, idx)
            X, y = X_full[idx], y_full[idx]

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
#                 print('UNIQ', np.unique(y))
#                 print('CLS', cls, sample)
                X_avg.append(X[sample])
                y_true.append(y[sample])
                
#             print(len(X_avg), X_avg[0].shape)

        return np.concatenate(tuple(X_avg)), np.concatenate(tuple(y_true))
    
    return sample_patient_data, len(data_files)


def train_pcas(sample_patient_data, X_train, n_samples_per_class):
    pcas = []

    for cls in range(5):
        X, _ = sample_patient_data(X_train, n_samples_per_class=n_samples_per_class, classes=[cls])

        model = PCA(n_components=7)
        model.fit(X)
        pcas.append(model)
#         print(model.n_components_, sum(model.explained_variance_ratio_))
        
    return pcas


def sample_classifier_data(sample_patient_data, patients, classif_train_thresh, n_samples_per_class):
    patients_train = patients[:classif_train_thresh]
    patients_test = patients[classif_train_thresh:]

    X_train, y_train = sample_patient_data(patients_train, n_samples_per_class=n_samples_per_class)
    X_valid, y_valid = sample_patient_data(patients_test, n_samples_per_class=n_samples_per_class)
    
    return X_train, X_valid, y_train, y_valid


def train_classifiers(pcas, X_train, X_valid, y_train, y_valid):
    # preprocess with pca
    Xs = []
    for X in (X_train, X_valid):
        X_pcas = pcas[0].transform(X)

        for pca in pcas[1:]:
            X_pca = pca.transform(X)
            X_pcas = np.hstack((X_pcas, X_pca))

        Xs.append(X_pcas)

    X_train_sparse, X_valid_sparse = Xs
    
    # define models
    knn = KNeighborsClassifier(7)
    svm = SVC(kernel="linear", C=0.025)
    mlp = MLPClassifier(alpha=1, max_iter=1000, activation='relu')

    classifiers = [knn, svm, mlp]

    # standardize data first
    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train_sparse)
    X_valid_std = scaler.transform(X_valid_sparse)

    results = []
    for model in classifiers:
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_valid_std)

#         print(model.__class__.__name__)
#         print(classification_report(y_valid, y_pred))
        d = classification_report(y_valid, y_pred, output_dict=True)
        results.append((d['accuracy'], d['macro avg']['f1-score']))
        
    return scaler, classifiers, list(zip(*results))


def fit_models(sample_patient_data, train_patients, n_samples_per_class, *, pca_classif_thresh=60, classif_train_thresh=48):
    patients_pca = train_patients[:pca_classif_thresh]
    patients_classif = train_patients[pca_classif_thresh:]
    
    pcas = train_pcas(sample_patient_data, patients_pca, n_samples_per_class)
    X_train, X_valid, y_train, y_valid = sample_classifier_data(sample_patient_data, patients_classif, classif_train_thresh, n_samples_per_class)
    
    scaler, classifiers, results = train_classifiers(pcas, X_train, X_valid, y_train, y_valid)
    return pcas, scaler, classifiers, results


def test_models(sample_patient_data, test_patients, pcas, scaler, classifiers):
    X_test, y_test = sample_patient_data(test_patients)
    
    X_pcas = pcas[0].transform(X_test)

    for pca in pcas[1:]:
        X_pca = pca.transform(X_test)
        X_pcas = np.hstack((X_pcas, X_pca))

    X_test_std = scaler.transform(X_pcas)
    
    results = []
    for model in classifiers:
        y_pred = model.predict(X_test_std)
        
#         print(model.__class__.__name__)
#         print(classification_report(y_test, y_pred))
        d = classification_report(y_test, y_pred, output_dict=True)
        results.append((d['accuracy'], d['macro avg']['f1-score']))
        
    return list(zip(*results))


def plot_results(n_samples, results):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Classification perfomance by number of sampled epochs')
    
    for i, (metric, metric_name) in enumerate([('acc', 'Accuracy'), ('f1', 'F1-score')]):
        valid_res = list(zip(*results[metric]['valid']))
        test_res = list(zip(*results[metric]['test']))
        
        axs[i].plot(n_samples, test_res[0], 'r-', label='knn_test')
        axs[i].plot(n_samples, test_res[1], 'g-', label='svm_test')
        axs[i].plot(n_samples, test_res[2], 'b-', label='mlp_test')
        axs[i].plot(n_samples, valid_res[0], 'r:', label='knn_valid')
        axs[i].plot(n_samples, valid_res[1], 'g:', label='svm_valid')
        axs[i].plot(n_samples, valid_res[2], 'b:', label='mlp_valid')
        axs[i].set(xlabel='Number of samples / class / patient', ylabel=metric_name)
        axs[i].legend()
        axs[i].grid()
    
    plt.show()
    
        
def run_experiment():
    rng = default_rng(42)
    
    eeg_sampler, data_count = make_data_sampler()

    TRAIN_SIZE = int(.8 * data_count)
    patient_permutation = rng.permutation(data_count)

    train_patients = patient_permutation[:TRAIN_SIZE]
    test_patients = patient_permutation[TRAIN_SIZE:]
    print(len(train_patients), len(test_patients))
    
    results = {
        'acc': {
            'valid': [],
            'test': [],
        },
        'f1': {
            'valid': [],
            'test': [],
        }
    }
    
#     n_samples_per_class = 10
#     sample_counts = list(range(1, 100))

    #n_samples = list(range(1, 90, 10))
    n_samples = list(range(1, 90, 1))
    
#     n_samples = list(range(1, 3))
    for train_samples_per_class in tqdm(n_samples):
        pcas, scaler, classifiers, (acc_valid, f1_valid) = fit_models(eeg_sampler, train_patients, train_samples_per_class)
        results['acc']['valid'].append(acc_valid)
        results['f1']['valid'].append(f1_valid)

        acc_test, f1_test = test_models(eeg_sampler, test_patients, pcas, scaler, classifiers)
        results['acc']['test'].append(acc_test)
        results['f1']['test'].append(f1_test)
        
    plot_results(n_samples, results)
    
#     print(results['acc'])
#     print(results['f1'])