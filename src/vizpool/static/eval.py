"""
Author : Hasanain Mehmood
Contact : hasanain@aicailber.com 
"""

import pickle
from array import array
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

plt.style.use('fivethirtyeight')


class Evaluation:
    """This class will provide the common Machine learning evaluation plots.

    Attributes:
            y_val (array): Validation or test data to be predicted.
    """

    def __init__(self, y_val: Union[pd.Series, array]):
        """Constructs all the necessary attributes for the instance of Evaluation class.

        Args:
            y_val (Union[pd.Series, array]): Validation or test data to be predicted.
        """
        self.y_val = y_val

    def confusion_matrix(self, y_predicted: array, target_names=None, normalize=True, width=20, height=20, title="Confusion Matrix") -> object:
        """This method will plot the confusion matrix with the following arguments provided:

        Args:
            y_predicted (array): An array of predicted values.
            target_names (list, optional): List of target names. Defaults to None.
            normalize (bool, optional): Whether to normalize the values or not. Defaults to True.
            width (int, optional): Width of the plot. Defaults to 20.
            height (int, optional): Height of the plot. Defaults to 20.
            title (str, optional): Title of the plot. Defaults to "Confusion Matrix".

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1.2)
        if target_names is not None and len(target_names) > 0:
            y_unique = target_names
        else:
            y_unique = list(set(self.y_val))
        classes = len(y_unique)
        cm = confusion_matrix(self.y_val, y_predicted)
        plt.figure(figsize=(width, height))
        if normalize:
            con = np.zeros((classes, classes))
            for x in range(classes):
                for y in range(classes):
                    con[x, y] = cm[x, y]/np.sum(cm[x, :])
            sns.heatmap(con, annot=True, fmt='.2', cmap='Blues',
                        xticklabels=y_unique, yticklabels=y_unique)
        else:
            sns.heatmap(cm, annot=True, cmap='Blues',
                        xticklabels=y_unique, yticklabels=y_unique, fmt='g')
        plt.title(title)
        return plt

    def feature_importance(self, estimator: object, X_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series, array],
                           width=15, height=10, title='Feature Importance') -> object:
        """This method will plot the chart for the feature importances with a trianed classifier of pipeline while following arguments being provided:

        Args:
            estimator (object): A trained estimator or a pipeline.
            X_train (pd.DataFrame): A pandas dataframe.
            y_train (Union[pd.DataFrame, pd.Series, array]): Class labels of the dataframe.
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 10.
            title (str, optional): Title of the plot. Defaults to 'Feature Importance'.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        estimator.fit(X_train, y_train)
        weights = None
        estimators = ['classifier', 'regressor', 'estimator']+[estimator]
        for estimator in estimators:
            if weights is None:
                try:
                    estimator_name = estimator[estimator].__class__.__name__
                except:
                    estimator_name = estimator.__class__.__name__
                try:
                    weights = estimator.coef_
                except:
                    try:
                        weights = estimator.feature_importances_
                    except:
                        try:
                            weights = estimator[estimator].coef_
                        except:
                            try:
                                weights = estimator[estimator].feature_importances_
                            except:
                                weights = None
        if weights is None:
            print(f"\n!!! No Coefficients to plot for {estimator_name}, skipping feature importance plot for {estimator_name}\n")
            return None
        weights = np.array(weights).flatten()
        try:
            weights_df = pd.DataFrame({
                'columns': X_train.columns,
                'weight': weights
            }).sort_values('weight', ascending=False)
        except:
            weights_df = pd.DataFrame({
                'columns': [i for i in range(len(weights))],
                'weight': weights
            }).sort_values('weight', ascending=False)
        weights_df["abs_value"] = weights_df["weight"].apply(lambda x: abs(x))
        weights_df["colors"] = weights_df["weight"].apply(
            lambda x: "green" if x > 0 else "red")
        weights_df = weights_df.sort_values("abs_value", ascending=False)

        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        sns.barplot(x="columns",
                    y="weight",
                    data=weights_df.head(30),
                    palette=weights_df.head(30)["colors"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
        ax.set_ylabel("Coef", fontsize=12)
        ax.set_xlabel("Feature Name", fontsize=10)
        plt.title(title, fontweight='bold', fontsize=15)
        return plt

    def auc_roc_plot(self, X_val: pd.DataFrame, classifiers: list, classifier_names: list, width=15, height=10, title='ROC Curve Analysis') -> object:
        """This method will plot the ROC curve analysis with the following parameters provided:

        Args:
            X_val (pd.DataFrame): Features of test or validation dataset.
            classifiers (list): List of classifiers or pipelines.
            classifier_names (list): List of classifier names.
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 10.
            title (str, optional): Title of the plot. Defaults to 'ROC Curve Analysis'.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        if len(np.unique(self.y_val)) > 2:
            print("AUC ROC Plot supports binary classification only, it is not available for Multiclass classification")
            return None
        result_dict = {'classifiers': [], 'fpr': [], 'tpr': [], 'auc': []}

        for model, clf_name in zip(classifiers, classifier_names):
            try:
                with open(model, 'rb') as f:
                    model = pickle.load(f)
            except:
                pass
            try:
                yproba = model.predict_proba(X_val)[::, 1]
            except AttributeError:
                try:
                    yproba = model._predict_proba_lr(X_val)[::, 1]
                except AttributeError:
                    yproba = None
            try:
                fpr, tpr, _ = roc_curve(self.y_val,  yproba)
                auc = roc_auc_score(self.y_val, yproba)

                result_dict['classifiers'].append(clf_name)
                result_dict['fpr'].append(fpr)
                result_dict['tpr'].append(tpr)
                result_dict['auc'].append(auc)
            except:
                pass
        result_table = pd.DataFrame(result_dict).set_index('classifiers')
        fig = plt.figure(figsize=(width, height))

        for i in result_table.index:
            plt.plot(result_table.loc[i]['fpr'],
                     result_table.loc[i]['tpr'],
                     label=f"{i}, AUC={result_table.loc[i]['auc']:.3f}")

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Flase Positive Rate", fontsize=15)
        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title(title, fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 13}, loc='lower right')
        return plt

    def residplot(self, y_predicted: Union[array, pd.Series], color="#ce1414", width=18, height=20) -> object:
        """This method will plot the residual plot with the following arguments provided: 

        Args:
            y_predicted (Union[pd.Series, array]): An array or Pandas series representing the predicted values.
            color (str, optional): Color of the plot. Defaults to "#ce1414".
            width (int, optional): Width of the plot. Defaults to 18.
            height (int, optional): Height of the plot. Defaults to 20.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        plt.figure(figsize=(width, height))
        sns.jointplot(x=self.y_val, y=y_predicted, kind='resid', color=color)
        return plt