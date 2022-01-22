import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import Counter
from sklearn.impute import SimpleImputer

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# upload data
PATH = r"ecg_data_all.csv"
df = pd.read_csv(PATH)
NAMES = df.columns


def prepareData(dataFrame):
    """The function is responsible for encoding class labels and inserting missing data

        Parameters
        ----------
        dataFrame : pandas.DataFrame
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
    class_mapping = {label: idx for idx, label in enumerate(np.unique(dataFrame["Class"]))}
    dataFrame["Class"] = dataFrame["Class"].map(class_mapping)

    # Insert missing data with imputation using mean. This is one of the data interpolation methods
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(dataFrame.values)
    idf = pd.DataFrame(imp.transform(dataFrame.values))
    idf.columns, idf.index = dataFrame.columns, dataFrame.index

    X, y = idf.iloc[:, :-1].values, idf.iloc[:, -1].values

    return X, y, class_mapping


def correlation_matrix_plot(data):
    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def scatter_matrix_plot(data):
    # Making scatter plot
    axes = pd.plotting.scatter_matrix(data, alpha=0.2)
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def histogram_plot(data):
    # Making histogram plot, you can visualize histogram for single feature by passing
    # sns.displot(data, x="HRV_MeanNN", hue="Class")
    data.hist()
    plt.show()


def density_plot(data):
    # Making density plot
    data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False)
    plt.show()


def box_whisker_plot(data):
    # Making box-whisker plot
    plt.tight_layout()
    data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
    plt.show()


def class_distribution(y, class_mapping):
    # Making class distribution plot
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
        plt.figure(2)
    plt.tight_layout()
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    bars = plt.barh([inv_class_mapping[x] for x in counter.keys()], counter.values())
    for bar in bars:
        width = bar.get_width()
        label_y = bar.get_y() + bar.get_height() / 2
        plt.text(width, label_y, s=f'{width}')
    plt.ylabel("Klasa")
    plt.xlabel("Ilość wektorów danych")
    plt.show()


def pairplot(data, method):
    # Making pairplot from seaborn module
    if method == "kde":
        sns.pairplot(data, kind="kde")
    elif method == "Class":
        sns.pairplot(data, hue="Class")
    else:
        pass
    plt.show()

# **********************************
# * Data visualization - an example *
# **********************************

# correlation_matrix_plot(df)
# scatter_matrix_plot(df)
# box_whisker_plot(df)
# density_plot(df)
# histogram_plot(df)
# pairplot(df[["QRS_Time", "HRV_MeanNN", "Class"]], "Class")

# X, y, class_mapping = prepareData(df)

# Distribution of classes
# class_distribution(y, class_mapping)
