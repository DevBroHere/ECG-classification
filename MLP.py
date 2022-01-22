import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU


LeakyReLU = LeakyReLU(alpha=0.1)

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

tf.get_logger().setLevel('ERROR')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def create_model(activation='selu', init_mode='he_uniform', neurons_enter=95, neurons_hidden=120, learning_rate=0.001,
                 shape=16):
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
    model.add(Dense(neurons_enter, input_shape=(None, shape),
                    activation=activation, kernel_initializer=init_mode, kernel_regularizer="l2"))
    model.add(Dense(neurons_hidden, activation=activation, kernel_initializer=init_mode))
    # 16 is the number of neurons present in the layer, the number corresponds to the number of classes
    model.add(Dense(16, kernel_initializer=init_mode, activation='softmax', kernel_regularizer="l2"))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


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
    Y : list
        List of labels values
    class_mapping : list
        Information about proper names of labels
    """
    # Class label encoding
    class_mapping = {label: idx for idx, label in enumerate(np.unique(df["Class"]))}
    df["Class"] = df["Class"].map(class_mapping)
    print(class_mapping)

    # Insert missing data with imputation using mean. This is one of the data interpolation methods
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(df.values)
    idf = pd.DataFrame(imp.transform(df.values))
    idf.columns, idf.index = df.columns, df.index

    X, y = idf.iloc[:, :-1].values, idf.iloc[:, -1].values

    return X, y, class_mapping


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


def grid_search(X, y, estimator, param_grid, scoring="f1_weighted", n_jobs=1,
                cv=StratifiedKFold(n_splits=5, shuffle=True), verbose=3, grid_mode="Normal", n_iter=60):
    """Performing grid search or random grid search.

    Parameters
    ----------
    X : list
        List of vectors (features)
    y : list
        List of labels values
    estimator
        model for estimation
    param_grid : dict
        Dictionary containing the values of hyperparameters
    scoring : str, optional
        What kind of score to use in evaluation (default is "f1_weighted")
    n_jobs : int, optional
        The number of jobs to run in parallel. Depending on number of cores
    cv : optional
        Method of splitting data for cross validation
    verbose : int, optional
        Controls what will be printed during procedure. It depends on the given integer:
        >1 : the computation time for each fold and parameter candidate is displayed;
        >2 : the score is also displayed;
        >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the
        computation.
    grid_mode : str, optional
        What type of grid search to use
    n_iter : int, optional
        How many iteration for randomized grid search (default is 60)
    """

    def grid_results(grid_result):
        """Collects the results of grid search and saves them in .txt format

        Parameters
        ----------
        grid_result : dict of numpy (masked) ndarrays
            Stores results of grid search.
        """

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_score = grid_result.best_score_
        best_params = grid_result.best_params_
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        msp_list = [["Best score: ", best_score, best_params]]
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            msp_list.append((mean, stdev, param))
        with open("Results_grid_search.txt", "w") as f:
            for s in msp_list:
                f.write(str(s) + "\n")

    scaler = preprocessing.StandardScaler()
    X_standarized = scaler.fit_transform(X)
    if grid_mode == "Normal":
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=n_jobs, cv=cv, verbose=verbose,
                            scoring=scoring)
        grid_result = grid.fit(X_standarized, y)
        grid_results(grid_result)
    elif grid_mode == "Random":
        grid = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_jobs=n_jobs, cv=cv,
                                  n_iter=n_iter,
                                  verbose=verbose)
        grid_result = grid.fit(X_standarized, y)
        grid_results(grid_result)


def check_model(model, X, y):
    """Collects the results of grid search and saves them in .txt format

    Parameters
    ----------
    model
        The trained model
    X : list
        List of vectors (features)
    y : list
        List of label values
    """

    scaler = preprocessing.StandardScaler()
    X_standarized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_standarized, y, test_size=0.2, stratify=y, shuffle=True)
    y_train = to_categorical(y_train, 16)
    y_test = to_categorical(y_test, 16)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    # Fit data to model
    history = model.fit(X_train, y_train, epochs=1000, batch_size=5, verbose=1,
                        callbacks=[callback], validation_split=0.2)
    plot_history(history)

    # Generate generalization metrics
    y_test_conf = np.argmax(y_test, axis=1)
    metrics = get_metrics(model, X_test, y_test_conf)
    print(metrics)


# upload data
df = pd.read_csv("ecg_data_all.csv")

# deletion of data on the least numerous label - VFL
indexNames = df[df["Class"] == "VFL"].index
df.drop(indexNames, inplace=True)

X, y, class_mapping = prepareData(df)

# create model for grid_search
model = KerasClassifier(build_fn=create_model, epochs=20, verbose=2, batch_size=5)

learning_rate = np.arange(0.001, 0.1, 100)
neurons_enter = np.arange(20, 500, 25)
neurons_hidden = np.arange(20, 1000, 50)
batch_size = np.arange(5, 100, 20)
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
activation = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
              'elu', 'exponential', LeakyReLU, "hard_sigmoid", "linear"]
param_grid = dict(learning_rate=learning_rate, neurons_enter=neurons_enter, neurons_hidden=neurons_hidden,
                  batch_size=batch_size)

# grid_search(X, y, model, param_grid, grid_mode="Random")
# check_model(model, X, y)
