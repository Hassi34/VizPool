"""
Author : Hasanain Mehmood
Contact : hasanain@aicailber.com 
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import get_cat_cols, get_num_cols

plt.style.use('fivethirtyeight')


class EDA:
    """This class have all the methods to perform complete Exploratory Data Analysis(EDA) on
       any Pandas dataframe passed as an argument on class initialization.

    Attributes:
            df (pd.DataFrame): Pandas dataframe to perform the EDA on.
    """

    def __init__(self, df: pd.DataFrame):
        """Constructs all the necessary attributes for the instance of EDA class.

        Args:
            df (pd.DataFrame): Pandas dataframe to perform the EDA on.
        """
        self.df = df

    def pie_bar(self, hue: str, kind="barh", color='red', width=18, height=6) -> object:
        """This method will plot side-by-side Pie and Bar chart with the following arguments provided:

        Args:
            hue (str): Name of the column to plot.
            kind (str, optional): Barchart kind. Defaults to "barh".
            color (str, optional): Font color. Defaults to 'red'.
            width (int, optional): Width of the plot. Defaults to 18.
            height (int, optional): Height of the plot. Defaults to 6.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        plt.figure(figsize=(width, height))
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       sharey=False, figsize=(width, height))
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

    def histogram(self, col_to_plot="default", hue=None, kde=True, width=25, height=15) -> object:
        """This method will plot the grid or single histogram with the following arguments provided:

        Args:
            col_to_plot (list, optional): A list of the names of numerical columns. Defaults to "default".
            hue (str, optional): A categorical column name to present hue in the data. Defaults to None.
            kde (bool, optional): kernel density estimate (KDE). Defaults to True.
            width (int, optional): Width of the plot. Defaults to 25.
            height (int, optional): Height of the plot. Defaults to 15.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        plt.figure(figsize=(width, height))
        for i in range((len(num_cols))):
            ax = plt.subplot(int(len(num_cols)/columns), columns, i+1)
            sns.histplot(
                data=self.df, x=num_cols[i], hue=hue, element="step", kde=kde)
        plt.tight_layout()
        return plt

    def boxplot(self, col_to_plot="default", hue=None, width=20, height=10) -> object:
        """This method will plot the grid or single box plot with the following parameters provided:

        Args:
            col_to_plot (list, optional): A list of the names of numerical columns. Defaults to "default".
            hue (str, optional): A categorical column name to present hue in the data. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 20.
            height (int, optional): Height of the plot. Defaults to 10.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        plt.figure(figsize=(width, height))
        for i in range((len(num_cols))):
            ax = plt.subplot(int(len(num_cols)/columns), columns, i+1)
            sns.boxplot(data=self.df, x=self.df[num_cols[i]], y=hue,
                        notch=True, showcaps=True,
                        flierprops={"marker": "x"},
                        boxprops={"facecolor": (.4, .6, .8, .5)},
                        medianprops={"color": "coral"},)
        plt.tight_layout()
        return plt

    def violinplot(self, hue: str, col_to_plot="default", inner="quart", width=15,
                   height=10, x_label_rotation=90, title="Violin Plot") -> object:
        """This method will plot the violinplot witht the following parameters provided:

        Args:
            hue (str): A categorical column name to present hue in the data.
            col_to_plot (str, optional): A list of the names of numerical columns. Defaults to "default".
            inner (str, optional): Representation of the datapoints in the violin interior, values could be: ['box','point','quart','stick']. Defaults to "quart".
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 10.
            x_label_rotation (int, optional): Rotation of the label. Defaults to 90.
            title (str, optional): Title of the plot. Defaults to "Violin Plot".

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
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
        plt.figure(figsize=(width, height))
        sns.violinplot(x="features", y="value", hue=y.name,
                       data=data, split=False, inner=inner)
        plt.title(title)
        plt.xticks(rotation=x_label_rotation)
        return plt

    def jointplot(self, x: str, y: str, kind="reg", color="#ce1414", width=18, height=20) -> object:
        """This method will plot the jointplot with the following arguments provided: 

        Args:
            x (str): Numeraical column name to be plotted on x-axis
            y (str): Numeraical column name to be plotted on y-axis
            kind (str, optional): Kind of chart to plot, available selections: ['scatter','kde','hist','hex','reg', 'resid']. Defaults to "reg".
            color (str, optional): Color of the plot. Defaults to "#ce1414".
            width (int, optional): Width of the plot. Defaults to 18.
            height (int, optional): Height of the plot. Defaults to 20.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        plt.figure(figsize=(width, height))
        sns.jointplot(data=self.df, x=x, y=y, kind=kind, color=color)
        return plt

    def corr_heatmap(self, col_to_plot='default', width=25, height=20, title="Correlation Plot") -> object:
        """This method will plot the correlation plot with the following arguments provided:

        Args:
            col_to_plot (list, optional): A list of names of numerical columns. Defaults to 'default'.
            width (int, optional): Width of the plot. Defaults to 25.
            height (int, optional): Height of the plot. Defaults to 20.
            title (str, optional): Title of the plot. Defaults to "Correlation Plot".

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        num_cols, columns = get_num_cols(self.df, col_to_plot)
        corr = self.df[num_cols].corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(width, height))
            ax = sns.heatmap(corr, mask=mask, vmax=1.0,
                             linewidths=.5, fmt='.1f', annot=True)
        plt.title(title)
        return plt

    def countplot(self, col_to_plot="default", hue=None, width=15, height=8) -> object:
        """This method will plot the count plot with the following arguments provided:

        Args:
            col_to_plot (list, optional): A list of the names of numerical columns. Defaults to "default".
            hue (_type_, optional): A categorical column name to present hue in the data. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 8.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        cat_cols, columns = get_cat_cols(self.df, col_to_plot)
        plt.figure(figsize=(width, height))
        for i in range((len(cat_cols))):
            ax = plt.subplot(int(len(cat_cols)/columns), columns, i+1)
            sns.countplot(data=self.df, x=cat_cols[i], hue=hue)
        plt.tight_layout()
        return plt

    def pairplot(self, hue: str, width=25, height=15) -> object:
        """This method will plot the pair plot with the following arguments provided:

        Args:
            hue (str): A categorical column name to present hue in the data.
            width (int, optional): Width of the plot. Defaults to 25.
            height (int, optional): Height of the plot. Defaults to 15.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        sns.set(font_scale=1)
        plt.figure(figsize=(width, height))
        sns.pairplot(data=self.df, hue=hue)
        return plt

    def barplot(self, y: str, hue=None, col_to_plot='default', estimator='mean', width=15, height=8) -> object:
        """This method will plot the bar chart with the following arguments provided:

        Args:
            y (str): Name of the column containing numerical data to plot
            hue (str, optional): A categorical column name to present hue in the data. Defaults to None.
            col_to_plot (list, optional): A list of column names to plot. Defaults to 'default'.
            estimator (str, optional): An aggregation function, available selections: [sum, mean, meadian, std, var]. Defaults to 'mean'.
            width (int, optional): Width of the plot. Defaults to 15.
            height (int, optional): Height of the plot. Defaults to 8.

        Returns:
            object: An Object which can be used to save or plot charts in any python application.
        """
        cat_cols, columns = get_cat_cols(self.df, col_to_plot)
        plt.figure(figsize=(width, height))
        for i in range((len(cat_cols))):
            ax = plt.subplot(int(len(cat_cols)/columns), columns, i+1)
            sns.barplot(
                data=self.df, x=cat_cols[i], y=y, estimator=estimator, hue=hue)
        plt.tight_layout()
        return plt
