import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .common import get_cat_cols, get_num_cols

plt.style.use('fivethirtyeight')


class EDA:
    def __init__(self, df):
        self.df = df

    def pie_bar(self, hue, kind="barh", color='red', fig_width=10, fig_height=6):
        plt.figure(figsize=(fig_width, fig_height))
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       sharey=False, figsize=(fig_width, fig_height))
        labels_list = []
        explode_list = []
        for i in range(self.df[hue].nunique()):
            labels_list.append(self.df[hue].value_counts().index[i])
            explode_list.append(0.01)
        ax1 = self.df[hue].value_counts().plot.pie(
            autopct="%1.0f%%", labels=labels_list, startangle=60,
            ax=ax1, wedgeprops={"linewidth": 1, "edgecolor": "k"}, explode=explode_list, shadow=True)
        ax1.set(title=f'Percentage in {hue}')

        ax2 = self.df[hue].value_counts().plot(kind=kind, ax=ax2)
        for i, j in enumerate(self.df[hue].value_counts().values):
            ax2.text(0.5, i, j, fontsize=20, color=color)
        ax2.set(title=f'Count in {hue}')
        return plt

    def histogram(self, col_to_plot="default", hue=None, title="Distribution Plot", fig_width=25, fig_height=15, kde=True):
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        plt.figure(figsize=(fig_width, fig_height))
        for i in range((len(num_cols))):
            ax = plt.subplot(int(len(num_cols)/columns), columns, i+1)
            sns.histplot(
                data=self.df, x=num_cols[i], hue=hue, element="step", kde=kde)
        plt.title(title)
        plt.tight_layout()
        return plt

    def boxplot(self, col_to_plot="default", hue=None, fig_width=25, fig_height=15):
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        plt.figure(figsize=(fig_width, fig_height))
        for i in range((len(num_cols))):
            ax = plt.subplot(int(len(num_cols)/columns), columns, i+1)
            sns.boxplot(data=self.df, y=self.df[num_cols[i]], x=hue,
                        notch=True, showcaps=True,
                        flierprops={"marker": "x"},
                        boxprops={"facecolor": (.4, .6, .8, .5)},
                        medianprops={"color": "coral"},)
        plt.tight_layout()
        return plt

    def violinplot(self, hue, col_to_plot="default", inner="quart", fig_width=15,
                   fig_height=10, x_label_rotation=90, title="Violin Plot"):
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        X = self.df[num_cols].drop(hue, axis=1, errors='ignore')
        y = self.df[hue]
        if hue in num_cols:
            columns_len = len(num_cols)-1
        else:
            columns_len = len(num_cols)
        data_n_2 = (X - X.mean()) / (X.std())
        data = pd.concat([y, data_n_2.iloc[:, 0:columns_len]], axis=1)
        data = data.fillna(method='ffill')
        data = pd.melt(data, id_vars=y.name,
                       var_name="features",
                       value_name='value')
        plt.figure(figsize=(fig_width, fig_height))
        sns.violinplot(x="features", y="value", hue=y.name,
                       data=data, split=False, inner=inner)
        plt.title(title)
        plt.xticks(rotation=x_label_rotation)
        return plt

    def joint_plot(self, var1, var2, kind="reg", color="#ce1414", title="Joint Plot", fig_width=18, fig_height=20):
        plt.figure(figsize=(fig_width, fig_height))
        sns.jointplot(self.df.loc[:, var1],
                      self.df.loc[:, var2], kind=kind, color=color)
        plt.title(title)
        return plt

    def corr_heatmap(self, col_to_plot='default', fig_width=25, fig_height=20, title="Correlation Plot"):
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        sns.set(font_scale=2)
        corr = self.df[num_cols].corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax = sns.heatmap(corr, mask=mask, vmax=1.0,
                             linewidths=.5, fmt='.1f', annot=True)
        plt.title(title)
        return plt

    def countplot(self, col_to_plot="default", hue=None, title="Count Plot", fig_width=25, fig_height=15):
        cat_cols, columns = get_cat_cols(self.df, col_to_plot)
        plt.figure(figsize=(fig_width, fig_height))
        for i in range((len(cat_cols))):
            ax = plt.subplot(int(len(cat_cols)/columns), columns, i+1)
            sns.countplot(data=self.df, x=cat_cols[i], hue=hue)
        plt.title(title)
        plt.tight_layout()
        return plt

    def pairplot(self, hue, title="Pair Plot", fig_width=25, fig_height=15):
        plt.figure(figsize=(fig_width, fig_height))
        sns.pairplot(data=self.df, hue=hue)
        plt.title(title)
        return plt

    def barplot(self, y, col_to_plot='default', estimator='mean', title="Bar Plot", fig_width=25, fig_height=20):
        cat_cols, columns = get_cat_cols(self.df, col_to_plot)
        plt.figure(figsize=(fig_width, fig_height))
        for i in range((len(cat_cols))):
            ax = plt.subplot(int(len(cat_cols)/columns), columns, i+1)
            sns.barplot(data=self.df, x=cat_cols[i], y=y, estimator=estimator)
        plt.tight_layout()
        plt.title(title, fontweight='bold', fontsize=15)
        return plt


class Evaluation:
    def __init__(self, y_val, y_predicted):
        self.y_val = y_val
        self.y_predicted = y_predicted

    def confusion_matrix(self, target_names=None, normalize=True, fig_width=20, fig_height=20, title="Correlation Plot"):
        '''
        This method takes true y values and predicted y values to plot a Confusion Matrix
        '''
        sns.set(font_scale=2)
        if target_names is not None and len(target_names) > 0:
            y_unique = target_names
        else:
            y_unique = list(set(self.y_val))
        classes = len(y_unique)
        cm = confusion_matrix(self.y_val, self.y_predicted)
        plt.figure(figsize=(fig_width, fig_height))
        if normalize:
            con = np.zeros((classes, classes))
            for x in range(classes):
                for y in range(classes):
                    con[x, y] = cm[x, y]/np.sum(cm[x, :])
            sns.heatmap(con, annot=True, fmt='.2', cmap='Blues',
                        xticklabels=y_unique, yticklabels=y_unique)
        else:
            #con = np.zeros((classes,classes))
            # for x in range(classes):
            #   for y in range(classes):
            #       con[x,y] = cm[x,y]
            sns.heatmap(cm, annot=True, cmap='Blues',
                        xticklabels=y_unique, yticklabels=y_unique)

        plt.title(title)
        return plt

    def feature_importance(self, classifier, X_train, y_train, fig_width=15, fig_height=10, title='Feature Importance'):
        '''
        This method takes a single classifier along with the X_train and y_train,
        fits the classifier/pipeline on the data and will returns the feature weights/importances
        '''

        classifier.fit(X_train, y_train)
        try:
            clf_name = classifier["classifier"].__class__.__name__
        except:
            clf_name = classifier.__class__.__name__
        try:
            weights = classifier.coef_
        except:
            try:
                weights = classifier.feature_importances_
            except:
                try:
                    weights = classifier["classifier"].coef_
                except:
                    try:
                        weights = classifier["classifier"].feature_importances_
                    except:
                        print(
                            f"\n!!! No Coefficients to plot for {clf_name}, skipping feature importance plot for {clf_name}")
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

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        sns.barplot(x="columns",
                    y="weight",
                    data=weights_df.head(30),
                    palette=weights_df.head(30)["colors"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15)
        ax.set_ylabel("Coef", fontsize=20)
        ax.set_xlabel("Feature Name", fontsize=12)
        plt.title(title, fontweight='bold', fontsize=15)
        return plt

    def auc_roc_plot(self, classifiers: list, X_val, y_val, fig_width=15, fig_height=10, title='ROC Curve Analysis'):
        '''
        This method will take a list of pipelines or classifiers, X_train, y_train, X_val and y_val. 
        It will return the AUC Plot for each the model provided in the input list.

        '''
        if len(np.unique(y_val)) > 2:
            return None
        result_dict = {'classifiers': [], 'fpr': [], 'tpr': [], 'auc': []}

        for model in classifiers:
            try:
                yproba = model.predict_proba(X_val)[::, 1]
            except AttributeError:
                try:
                    yproba = model._predict_proba_lr(X_val)[::, 1]
                except AttributeError:
                    return None
            fpr, tpr, _ = roc_curve(y_val,  yproba)
            auc = roc_auc_score(y_val, yproba)
            try:
                clf_name = model["classifier"].__class__.__name__
            except:
                clf_name = model.__class__.__name__
            result_dict['classifiers'].append(clf_name)
            result_dict['fpr'].append(fpr)
            result_dict['tpr'].append(tpr)
            result_dict['auc'].append(auc)

        result_table = pd.DataFrame(result_dict).set_index('classifiers')
        fig = plt.figure(figsize=(fig_width, fig_height))

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
