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
from sklearn.inspection import permutation_importance
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

    def feature_importance(self, estimator: object, X_val: object, pipeline=False,
                           width: int = 15, height: int = 10) -> object:
        """This method will plot the chart for the feature importances with a trianed classifier of pipeline with following arguments being provided:

        Args:
            estimator (object): A trained estimator or a pipeline.
            X_test (object): A pandas dataframe.
            pipeline (bool, optional): Whether the estimator is pipeline object or not. Defaults to False.
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 10.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        weights = None
        if pipeline:
            feature_names = [str(feature).split("__")[-1]
                             for feature in estimator[:-1].get_feature_names_out()]
            try:
                result = permutation_importance(
                    estimator, X_val, self.y_val, n_repeats=10, random_state=42, n_jobs=-1)
            except:
                pass
            for model in ['classifier', 'regressor', 'estimator']:
                if weights is None:
                    try:
                        estimator_name = estimator[model].__class__.__name__
                    except:
                        pass
                    try:
                        weights = estimator[model].coef_[0]
                    except:
                        try:
                            weights = estimator[model].feature_importances_
                        except:
                            weights = None
        else:
            estimator_name = estimator.__class__.__name__
            feature_names = X_val.columns.tolist()
            result = permutation_importance(
                estimator, X_val, self.y_val, n_repeats=10, random_state=42, n_jobs=-1)
            try:
                weights = estimator.coef_[0]
            except:
                try:
                    weights = estimator.feature_importances_
                except:
                    weights = None
        if weights is None:
            print(
                f"\n!!! Feature importance plot is not available for {estimator_name}, skipping feature importance for {estimator_name}...\n")
            return None
        weights = np.array(weights).flatten()
        try:
            weights_df = pd.DataFrame({
                'columns': feature_names,
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

        fig = plt.figure(figsize=(width, height))
        plt.subplot(1, 2, 1)
        sns.barplot(y="columns",
                    x="weight",
                    data=weights_df.head(30),
                    palette=weights_df.head(30)["colors"])
        plt.yticks(rotation=45)
        plt.ylabel("Feature Name", fontsize=12)
        plt.xlabel("Feature Importance", fontsize=10)
        plt.title("Feature Importance", fontweight='bold', fontsize=15)
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=(X_val.columns)[sorted_idx],
        )
        plt.yticks(rotation=45)
        plt.xlabel("Feature Importance", fontsize=10)
        plt.title("Permutation Importance (test set)",
                  fontweight='bold', fontsize=15)
        fig.tight_layout()
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
