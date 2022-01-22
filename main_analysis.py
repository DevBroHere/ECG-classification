import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

PATH = r"ecg_data_all.csv"
df = pd.read_csv(PATH)
NAMES = df.columns


def prepareData(df):
    """The function is responsible for encoding class labels and inserting missing data

    Parameters
    ----------
    df : pandas.DataFrame
        The type of data containing the feature space with classes

    Returns
    -------
    X : list
        List of vectors (features)
    y : list
        List of labels values
    class_mapping : list
        Information about proper names of labels
    """

    # Class label encoding
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df["Class"]))}
    df["Class"] = df["Class"].map(class_mapping)
    # print(class_mapping)

    # Insert missing data with imputation using mean. This is one of the data interpolation methods
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(df.values)
    idf = pd.DataFrame(imp.transform(df.values))
    idf.columns, idf.index = df.columns, df.index

    X, y = idf.iloc[:, :-1].values, idf.iloc[:, -1].values

    return X, y, class_mapping


def confusion_matrix_plot(cm, title=None, class_mapping=None, save=False, show=True):
    """Plot confusion matrix (if necessary) and save it.

    Parameters
    ----------
    cm : ndarray
        The confusion matrix array
    title : str
        The name of saved png file
    class_mapping : dict
        Contains label names
    save : bool, optional
        If true then save confusion matrix. Otherwise don't save it
    show : bool, optional
        If true then show confusion matrix. Otherwise don't show it
    """

    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
    plt.title(title)
    if class_mapping:
        inv_class_mapping = {v: k for k, v in class_mapping.items()}
        ax.set_xticks(range(0, len(inv_class_mapping.values())))
        ax.set_yticks(range(0, len(inv_class_mapping.values())))
        ax.set_xticklabels(list(inv_class_mapping.values()))
        ax.set_yticklabels(list(inv_class_mapping.values()))
        plt.xticks(rotation=45)
    plt.xlabel("Przewidywana etykieta")
    plt.ylabel("Rzeczywista etykieta")
    if show:
        plt.show()
    if save:
        fig.savefig(title + '.png', format="png")
        plt.close(fig)


def get_metrics(model, X_test, y_test):
    """Evaluation of the model with selected performance metrics

    Parameters
    ----------
    model
        The trained model
    X_test
        Test X data (features)
    y_test
        Test y data (labels)

    Returns
    -------
    dictionary
        A dictonary containing all selected metrics: accuracy, weighted recall, weighted f1, weighted auc, error
    """

    model_pred = model.predict(X_test)
    model_prob = model.predict_proba(X_test)  # propability is for ROC AUC score
    model_accuracy = accuracy_score(y_test, model_pred)
    model_recall = recall_score(y_test, model_pred, average="weighted")
    model_f1 = f1_score(y_test, model_pred, average='weighted')
    model_auc = roc_auc_score(y_test, model_prob, average="weighted", multi_class="ovr")
    ERR_count = 0
    for row_index, (input, prediction, label) in enumerate(zip(X_test, model_pred, y_test)):
        if prediction != label:
            ERR_count += 1
    print(model.__class__.__name__)
    print('Accuracy: ', "{:.2f}%".format(model_accuracy * 100))
    print("Recall: ", "{:.2f}%".format(model_recall * 100))
    print('F1: ', "{:.2f}%".format(model_f1 * 100))
    print("ROC AUC score:", "{:.2f}%".format(model_auc * 100))
    print("ERR:", "{0}/{1}".format(ERR_count, len(y_test)))
    # if showReport:
    # print(classification_report(y_test, model_pred, target_names=list(class_mapping.keys())))
    return {"Accuracy": model_accuracy, "Recall": model_recall, "F1": model_f1, "AUC": model_auc, "ERR": ERR_count}


def PCA_explained_variance_show(X_train):
    """Allows displaying the variance as a function of the number of features.

    Parameters
    ----------
    X_train
        Train X data (features)
    """

    pca = PCA().fit(X_train)

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 17, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Ilość składowych')
    plt.xticks(np.arange(0, 17, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Wariancja skumulowana (%)')
    plt.title('Liczba składników potrzebnych do wyjaśnienia wariancji')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% próg odcięcia', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()


def calculate_ERR(clf, X, y):
    # Calculates error value
    ERR_count = 0
    y_pred = clf.predict(X)
    for row_index, (prediction, label) in enumerate(zip(y_pred, y)):
        if prediction != label:
            ERR_count += 1
    return ERR_count


def select_n_components(var_ratio, goal_var: float) -> int:
    # Defining the initial value of the explained variance
    total_variance = 0.0

    # Defining the initial number of features
    n_components = 0

    # For each presented variance of the highlighted individual features:
    for explained_variance in var_ratio:

        # Add the explained variance to the total variance
        total_variance += explained_variance

        # Add the value one to the number of components
        n_components += 1

        # When the desired goal of explained variance is achieved
        if total_variance >= goal_var:
            # Ending the loop
            break

    # Return of the number of components
    return n_components


def plot_history(history):
    """Displaying efficiency and cost as a function of the epoch.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Contains the recorded values of accuracy and loss per epoch
    """

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for i in loss_list:
        plt.plot(epochs, history.history[i], 'b',
                 label='Training loss (' + str(str(format(history.history[i][-1], '.5f')) + ')'))
    for i in val_loss_list:
        plt.plot(epochs, history.history[i], 'g',
                 label='Validation loss (' + str(str(format(history.history[i][-1], '.5f')) + ')'))
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.figure(2)
    for i in acc_list:
        plt.plot(epochs, history.history[i], 'b',
                 label='Training accuracy (' + str(format(history.history[i][-1], '.5f')) + ')')
    for i in val_acc_list:
        plt.plot(epochs, history.history[i], 'g',
                 label='Validation accuracy (' + str(format(history.history[i][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def print_distribution(y):
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))


def make_cross_validation(X, y, n_splits, classifiers, reduction_methods, oversampling_methods):
    """Perform cross validation on given models, reduction and oversampling methods.

    Parameters
    ----------
    X : array
        List of vectors (features)
    y : array
        List of labels values
    n_splits : int
        How many parts should the data be divided into
    classifiers : list
        List of classifiers
    reduction_methods : list
        List of reduction methods
    oversampling_methods : list
        List of oversampling methods
    """

    models_df = pd.DataFrame()
    stratKFold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # for each model type, we change subsequent column transformers
    for reduction in reduction_methods:
        for oversampling in oversampling_methods:
            for model in classifiers:
                classification_report_per_fold = []
                conf_matrix = np.zeros((16, 16))
                fold_no = 0
                n_components = 16
                for train_index, test_index in stratKFold.split(X, y):
                    fold_no += 1
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # print_distribution(y_train)
                    if reduction is None:
                        pass
                    elif reduction.__class__.__name__ == "PCA":
                        # PCA_explained_variance_show(X_train)
                        reduction.fit(X_train)
                        X_train = reduction.transform(X_train)
                        X_test = reduction.transform(X_test)
                        n_components = reduction.n_components_
                    elif reduction.__class__.__name__ == "LinearDiscriminantAnalysis":
                        reduction.fit(X_train, y_train)
                        lda_var_ratios = reduction.explained_variance_ratio_
                        n_components = select_n_components(lda_var_ratios, 0.95)
                        reduction = LinearDiscriminantAnalysis(n_components=n_components)
                        reduction.fit(X_train, y_train)
                        X_train = reduction.transform(X_train)
                        X_test = reduction.transform(X_test)
                    if oversampling is None:
                        pass
                    else:
                        X_train, y_train = oversampling.fit_resample(X_train, y_train)
                    if model.__class__.__name__ == "KerasClassifier":
                        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
                        model = KerasClassifier(
                            build_fn=lambda: create_model(activation="tanh", init_mode="he_normal", shape=n_components),
                            epochs=100,
                            batch_size=25,
                            verbose=1, callbacks=[callback])
                        y_train = to_categorical(y_train, 16)
                        y_test = to_categorical(y_test, 16)
                        # Measuring time of training
                        start_time = time.time()
                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
                        end_time = time.time()
                        # plot_history(history)
                        # y_pred = model.predict(X_test)
                        # y_pred_conf = np.argmax(y_pred, axis=1)
                        y_test_conf = np.argmax(y_test, axis=1)
                        conf_matrix += confusion_matrix(y_test_conf, model.predict(X_test))
                        metrics = get_metrics(model, X_test, y_test_conf)
                        classification_report_per_fold.append(classification_report(y_test_conf, model.predict(X_test),
                                                                                    target_names=list(
                                                                                        class_mapping.keys())))
                    else:
                        # trenujemy tak przygotowany model (cały pipeline) mierząc ile to trwa
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        end_time = time.time()
                        conf_matrix += confusion_matrix(y_test, model.predict(X_test))
                        metrics = get_metrics(model, X_test, y_test)
                        classification_report_per_fold.append(
                            classification_report(y_test, model.predict(X_test),
                                                  target_names=list(class_mapping.keys())))
                    # zbieramy w dict parametry dla Pipeline i wyniki
                    param_dict = {
                        'fold': fold_no,
                        'Oversampling': oversampling.__class__.__name__,
                        'Reduction': reduction.__class__.__name__,
                        'n_components': n_components,
                        'test size': len(y_test),
                        'model': model.__class__.__name__,
                        "accuracy": metrics["Accuracy"],
                        "recall": metrics["Recall"],
                        "f1": metrics["F1"],
                        "auc": metrics["AUC"],
                        "err": metrics["ERR"],
                        'time_elapsed': end_time - start_time
                    }
                    models_df = models_df.append(pd.DataFrame(param_dict, index=[0]))
                confusion_matrix_plot(conf_matrix.astype("int32"),
                                      "Confusion_matrix" +
                                      model.__class__.__name__ + " - " + reduction.__class__.__name__ + " - "
                                      + oversampling.__class__.__name__,
                                      class_mapping, save=True, show=False)
                with open(
                        "Results_classification_report" + model.__class__.__name__ + " - " +
                        reduction.__class__.__name__ + " - "
                        + oversampling.__class__.__name__ + ".txt", "w") as f:
                    for s in classification_report_per_fold:
                        f.write(str(s) + "\n")

    models_df.reset_index(drop=True, inplace=True)
    models_df.to_excel("Models_comparison_best_models.xlsx")


def create_model(activation='selu', init_mode='he_uniform', neurons_enter=95, neurons_hidden=120, shape=16,
                 learning_rate=0.01):
    """The function creates an MLP neural network with the given hyperparameters.
    It is also used for the purpose of grid search and random grid search.

    Parameters
    ----------
    activation : str
        Given activation function
    init_mode : str
        Given initialization mode
    neurons_enter : int
        Given nubmer of neurons in input layer
    neurons_hidden : int
        Given number of neurons in hidden layer
    shape : int
        Number of inputs/features
    learning_rate : float
        Given learning rate

    Returns
    -------
    model
        The keras MLP model
    """

    # create model
    model = Sequential()
    # model.add(tf.keras.layers.Dropout(0.1, input_shape=(None, shape)))
    model.add(Dense(neurons_enter, input_shape=(None, shape),
                    activation=activation, kernel_initializer=init_mode, kernel_regularizer="l2"))
    model.add(Dense(neurons_hidden, activation=activation, kernel_initializer=init_mode))
    # 16 is the number of neurons present in the layer, the number corresponds to the number of labels
    model.add(Dense(16, kernel_initializer=init_mode, activation='softmax', kernel_regularizer="l2"))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# *****************
# * Preprocessing *
# *****************

# deletion of data on the least numerous label - VFL
indexNames = df[df["Class"] == "VFL"].index
df.drop(indexNames, inplace=True)

# Preparation of data for the learning process: class coding, imputation
X, y, class_mapping = prepareData(df)

# Standaryzacja i skalowanie wektorów uczących
scaler = preprocessing.StandardScaler()
X_standarized = scaler.fit_transform(X)
# print("Mean: ", round(X_standarized[:,0].mean()))
# print("Std: ", round(X_standarized[:,0].std()))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
modelMLP = KerasClassifier(build_fn=lambda: create_model(activation="tanh", init_mode="he_normal"), epochs=100,
                           batch_size=5,
                           verbose=1, callbacks=[callback])

classifiers = [
    DummyClassifier(strategy='stratified'),
    RandomForestClassifier(),
    SVC(probability=True),
    xgb.XGBClassifier(),
    MLPClassifier(),
    modelMLP
]

oversampling_methods = [
    None,
    SMOTE(),
    SMOTENC([1]),
    BorderlineSMOTE(),
    SVMSMOTE(),
    RandomOverSampler(),
    SMOTEN(),
    SMOTEENN(),
    SMOTETomek()
]

reduction_methods = [
    None,
    LinearDiscriminantAnalysis(n_components=None),
    PCA(n_components=0.95)
]

# PCA_explained_variance_show(X_standarized)
make_cross_validation(X_standarized, y, n_splits=5, classifiers=classifiers, oversampling_methods=oversampling_methods,
                      reduction_methods=reduction_methods)
