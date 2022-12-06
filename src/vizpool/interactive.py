"""
Author : Hasanain Mehmood
Contact : hasanain@aicailber.com 
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EDA:
    """This class has all the methods to perform complete Exploratory Data Analysis(EDA) on
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

    def barchart(self, values: str, categories: str, aggfunc: str='sum', data_labels: bool=True, title: str='Bar Chart', sort_by: str=None, ascending: bool=True,
                 orientation: str='h', texposition: str="auto", text_color: str='white', width: int=600, height: int=500) -> object:       
        """This method will plot a Bar chart with the following arguments provided:

        Args:
            values (str): Name of the column having numerical data to plot against the categories.
            categories (str): Name of the column having the list of categories as elements.
            aggfunc (str, optional): Aggregation function. Defaults to 'sum'.
            data_labels (bool, optional): Show data labels. Defaults to True.
            title (str): The title of the chart. Defaults to 'Bar Chart'.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            orientation (str): To plot the chart horizontly or vertically. Defaults to 'h'.
            texposition (str): Position of the text on chart. Defaults to "auto".
            text_color (str): Text color. Defaults to 'white'.
            width (int, optional): The width of the visualization. Defaults to 600.
            height (int, optional): The height of the visualization. Defaults to 500.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application
        """
        df = self.df.groupby([categories], as_index=False).agg(
            value=pd.NamedAgg(values, aggfunc=aggfunc)
        )
        df.columns = [categories, values]
        if sort_by:
            df = df.sort_values(sort_by, ascending = ascending)
        if data_labels:
            text = [round(val, 0) for val in df[values]]
        else: text = None
        if orientation == 'h':
            x = df[values]
            y = df[categories]
        else:
            y = df[values]
            x = df[categories]
        fig = go.Figure(go.Bar(
            x=x,
            y=y,
            orientation=orientation,
            text=text, textposition=texposition, textfont=dict(color=text_color),))

        fig.update_layout(barmode='stack', xaxis={
            'categoryorder': 'total descending'})
        fig.update_layout(autosize=False, width=width,
                          height=height,title=title, title_x=0.5)
        return fig

    def heatmap(self, index: str, columns: str, values: str, aggfunc: str='mean', data_labels: bool=True, x_label: str=None, y_label: str=None,
                 color_label: str=None, width: int=600, height: int=450, title: str='Heatmap') -> object:    
        """Thi method will plot a heatmap with the following arguments provided:

        Args:
            index (str): Name of the column to be set as index.
            columns (str): Name of the column to be set as column.
            values (str): Column name having numerical values.
            aggfunc (str, optional): Type of aggregation to be applied.
            data_labels (bool, optional): Show data labels. Defaults to True.
            x_label (str, optional): Label for x-axis. Defaults to None.
            y_label (str, optional): Label for y-axis. Defaults to None.
            color_label (str, optional): Color of the label. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the visualization. Defaults to Heatmap.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application
        """
        df_pivot = self.df[[index, columns, values]].pivot_table(
            index=index, columns=columns, aggfunc=aggfunc, fill_value=0)
        fig = px.imshow(df_pivot.values,
                        labels=dict(x=x_label, y=y_label, color=color_label),
                        x=[x[1] for x in df_pivot.columns],
                        y=df_pivot.index.tolist(), text_auto=data_labels
                        )
        fig.update_traces(dict(showscale=False,
                               coloraxis=None,
                               colorscale='bluered'), selector={'type': 'heatmap'})
        fig.update_xaxes(side="top", title=title)
        fig.update_layout(autosize=False, width=width, height=height,)
        return fig

    def stack_or_group_chart(self, categories: str, values: list, barmode: str="stack", orientation: str='v', sort_by: str=None, ascending: bool=True, unit: str=None, unit_position : str="after",
                             data_labels: bool=True, texposition: str='inside', aggfunc: str="mean", drop_column: str=None, text_color: str='white', width: int=600, height: int=450, title: str="Stack or Group Chart") -> object:
        """This method will plot a stack or group chart using the following arguments provided:

        Args:
            categories (str): Categorical Column
            values (list): list of numerical columns to plot
            barmode (str, optional): The position of the bars for the relavent data. Defaults to "stack".
            orientation (str, optional): Orientation of the graph could be vertical or horizontal. Defaults to 'v'.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            data_labels (bool, optional): Show data labels. Defaults to True.
            unit (str, optional): Unit to be displayed with the datalabels. Defaults to None.
            unit_position (str, optional): Position of the units of data labels, which could be before or after the label values. Defaults to 'after'.
            texposition (str, optional): Position of the text labels on the plot. Defaults to 'inside'.
            aggfunc (str, optional): Aggregation function. Defaults to "mean".
            drop_column (str, optional): A column to drop from the input data. 
                                        This could be useful for sorting the data on specific column and then droping it off. Defaults to None.
            text_color (str, optional): Color of text. Defaults to 'white'.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to "Stack or Group Chart".

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application
        """
        df = self.df[[categories]+values].groupby(
            [categories], as_index=False).agg(aggfunc, numeric_only=True)
        if sort_by:
            df = df.sort_values(sort_by, ascending = ascending)
        if drop_column:
            df = df.drop(columns = [drop_column])
            values.remove(drop_column)

        if unit == None:
            unit = " "
        else:
            unit = " "+unit

        data = []
        if orientation == 'v':
            for i in range(len(values)):
                if data_labels:
                    try:
                        text = [str(int(val))+unit if unit_position == 'after' else unit+' '+str(int(val)) for val in df[values[i]]]
                    except ValueError:
                        text = [str(round(float(val),0))+unit if unit_position == 'after' else unit+' '+str(round(float(val),0)) for val in df[values[i]]]
                else: text = None
                data.append(go.Bar(name=values[i], x=df[categories], y=df[values[i]], orientation=orientation,
                                   text=text, textposition=texposition, textfont=dict(color=text_color),))
        else:
            for i in range(len(values)):
                if data_labels:
                    try:
                        text = [str(int(val))+unit if unit_position == 'after' else unit+' '+str(int(val)) for val in df[values[i]]]
                    except ValueError:
                        text = [str(round(float(val),0))+unit if unit_position == 'after' else unit+' '+str(round(float(val),0)) for val in df[values[i]]]
                else: text = None
                data.append(go.Bar(name=values[i], y=df[categories], x=df[values[i]], orientation=orientation,
                                   text=text, textposition=texposition, textfont=dict(color=text_color),))
        fig = go.Figure(data=data)
        fig.update_layout(barmode=barmode, autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def piechart(self, categories: str, values: str, width: int=600, height: int=500, title: str='Pie Chart') -> object:
        """This method will plot a pie chart using with the following arguments provided:

        Args:
            categories (str): Categorical column name
            values (str): Numerical Column name
            width (int, optional): width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 500.
            title (str, optional): Title of the plot. Defaults to 'Pie Chart'.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application
        """
        fig = px.pie(data_frame=self.df, values=values, names=categories,
                     color=categories,
                     title=title,
                     template='presentation',
                     width=width,
                     height=height,
                     hole=0.5,
                     )
        return fig

    def area_chart(self, categories: str, values: str, aggfunc: str='mean', data_labels: bool=True, sort_by: str=None, ascending: bool=True,
                    unit: str=None, unit_position : str="after", width: int=600, height: int=450,  title: str="Area Chart") -> object:
        """This method will plot the area chart with the following arguments provided:

        Args:
            categories (str): categorical column name.
            values (str): column name for values to plot.
            aggfunc (str, optional): Aggregation function. Defaults to 'mean'.
            data_labels (bool, optional): Show data labels. Defaults to True.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            unit (str, optional): Unit to be displayed with the datalabels. Defaults to None.
            unit_position (str, optional): Position of the units of data labels, which could be before or after the label values. Defaults to 'after'.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to "Area Chart".

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        if sort_by == values:
            df = self.df.groupby(categories, as_index=False)\
            .agg(aggfunc, numeric_only=True)[[categories]+[values]]
            if sort_by:
                df = df.sort_values(sort_by, ascending = ascending)

        elif (sort_by != values) & (sort_by is not None):
            df = self.df[[categories, values, sort_by]].groupby(categories, as_index=False).agg(
            value=pd.NamedAgg(values, aggfunc=aggfunc),
            month_no=pd.NamedAgg(sort_by, aggfunc=pd.Series.mode)
            ).sort_values("month_no", ascending = ascending)
            df.columns = [categories]+ [values] + [sort_by]
            
        else:
            df = self.df.groupby(categories, as_index=False)\
                .agg(aggfunc, numeric_only=True)[[categories]+[values]]

        if data_labels:
            if unit == None:
                unit = " "
            else:
                unit = " "+unit
            try:
                text = [str(int(val))+unit if unit_position == 'after' else unit+' '+str(int(val)) for val in df[values]]
            except ValueError:
                text = [str(round(float(val),0))+unit if unit_position == 'after' else unit+' '+str(round(float(val),0)) for val in df[values]]
        else: text = None

        fig = px.area(
            df,
            x=df[categories],
            y=df[values],
            text=text, title=title
        )
        fig.update_traces(mode='markers+lines+text',
                          textfont_size=12, textposition="top left")
        fig.update_layout(autosize=False, width=width,
            height=height,title=title, title_x=0.5)
        return fig

    def bar_line(self, categories: str, values: list, aggfunc: list=['mean', 'sum', 'max'], data_labels: bool=True, sort_by: str=None, ascending: bool=True, legends: list=None,
                drop_column: str=None, width: int=600, height: int=450,title: str="Bar Line Chart", texposition: str="auto", text_color: str="white", round_decimal: int=0) -> object:
        """This method will plot a bar line chart with the following arguments provided:

        Args:
            categories (str): Name of the categorical column.
            values (list): A list of the names of numerical columns.
            aggfunc (list, optional): List of aggregation functions. Defaults to ['mean', 'sum', 'max'].
            data_labels (bool, optional): Show data labels. Defaults to True.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            legends (list, optional): List containing the custom legends as elements. Defaults to None.
            drop_column (str, optional): A column to drop from the input data. 
                            This could be useful for sorting the data on specific column and then droping it off. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to "Bar Line Chart".
            texposition (str, optional): Position of the data labels on the plot. Defaults to "auto".
            text_color (str, optional): Color of the text. Defaults to "white".
            round_decimal (int, optional): Decimal points to round values on. Defaults to 0.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        if len(values) == 2:
            df = self.df[[categories]+values].groupby([categories], as_index=False).agg(
                val1=pd.NamedAgg(values[0], aggfunc=aggfunc[0]),
                val2=pd.NamedAgg(values[1], aggfunc=aggfunc[1])
            )
            df.columns = [categories]+values
            if sort_by:
                df = df.sort_values(sort_by, ascending = ascending)
        if len(values) > 2:
            df = self.df[[categories]+values].groupby([categories], as_index=False).agg(
            val1=pd.NamedAgg(values[0], aggfunc=aggfunc[0]),
            val2=pd.NamedAgg(values[1], aggfunc=aggfunc[1]),
            val3=pd.NamedAgg(values[2], aggfunc=aggfunc[2])
            )
            df.columns = [categories]+values
            if sort_by:
                df = df.sort_values(sort_by, ascending = ascending)
            if drop_column and drop_column in df.columns.tolist():
                df = df.drop(columns = [drop_column])
                values.remove(drop_column)
        if data_labels:
            if round_decimal > 0:
                line_data_labels = [round(val, round_decimal)
                                    for val in df[values[0]]]
                bar_data_labels = [round(val, round_decimal)
                                for val in df[values[1]]]
            elif round_decimal == 0:
                line_data_labels = [int(val) for val in df[values[0]]]
                bar_data_labels = [int(val) for val in df[values[1]]]
        else: 
            line_data_labels = None
            bar_data_labels = None

        if legends is None:
            legends = values
    
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(name=legends[0], x=df[categories], y=df[values[0]],
                       text=line_data_labels, textposition='top right',
                       mode='lines+markers+text', marker=dict(color='#5D69B1', size=7)
                       ))
        fig.add_trace(
            go.Bar(name=legends[1], x=df[categories], y=df[values[1]],
                   text=bar_data_labels, textposition=texposition, textfont=dict(color=text_color),)
        )
        # yanchor="top",y=0.99 ,x=0.01
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def histogram(self, values: str, color: str=None, marginal: str=None, hover_data: list=None, title: str='Histogram', width: int=800, height: int=450) -> object:
        """This method with plot a histogram with following parameters provided:

        Args:
            values (str): Name of the numerical column to plot.
            color (str, optional): Name of the column to differentiate data points on color. Defaults to None.
            marginal (str, optional): Type of visual to plot as marginal. Defaults to None.
            hover_data (list, optional): A list of column names to display the data on hover to the marginal visual. Defaults to None.
            title (str, optional): Title of the plot. Defaults to 'Histogram'.
            width (int, optional): Width of the plot. Defaults to 800.
            height (int, optional): Height of the plot. Defaults to 450.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        fig = px.histogram(self.df, x=values, y=values, color=color,
                           marginal=marginal, histfunc='count', hover_data=hover_data)
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def distplot(self, values: list, width: int=800, height: int=450, title: str='Distribution Plot') -> object:
        """Thi method will plot the distribution plot with the following parameters provided:

        Args:
            values (list): A list of the names of the numerical columns to plot.
            width (int, optional): Width of the plot. Defaults to 800.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to 'Distribution Plot'.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        df = self.df[values]
        fig = ff.create_distplot(
            [df[c] for c in df.columns], df.columns, bin_size=.25)
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def combined_corr(self, x_values: str, y_values: str, color: str=None, hover_name: str=None, size: str=None,
                    width: int=600, height: int=450, title: str="Combined Correlation Plot") -> object:
        """This method will plot the combined correlation plot using the following arguments:

        Args:
            x_values (str): Name of the column to plot against x_axis.
            y_values (str): Name of the column to plot against y_axis.
            color (str, optional): Name of the column to differentiate data points on color. Defaults to None.
            hover_name (str, optional): Column name to display as hover name. Defaults to None.
            size (str, optional): Name of the column to be presented as the size of the data point. Defaults to None.
            width (int, optional): With of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to "Combined Correlation Plot".

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        fig = px.scatter(self.df, x=x_values, y=y_values, color=color,
                         marginal_x="box", marginal_y='violin', hover_name=hover_name,
                         size=size)
        fig.update_layout(autosize=False, width=width,
            height=height,title=title, title_x=0.5)
        return fig

    def multivar_bubble_chart(self, categories: list, values: list, aggfunc: str="mean", hover_name: str=None, sort_by: str=None, ascending: bool=True,
                              width: int=600, height: int=450, title: str="Bubble Chart") -> object:
        """This method will plot a multivariate bubble chart with the following arguments provided:

        Args:
            categories (list): List of the names of categorical columns to plot.
            values (list): List of the names of values columns to plot.
            aggfunc (str, optional): Aggregation function. Defaults to "mean".
            hover_name (str, optional): Name of the category to be displayed on hover. Defaults to None.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            width (int, optional): With of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to "Bubble Chart".

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        df = self.df.groupby(categories, as_index=False).agg(
            aggfunc, numeric_only=True)[categories+values]
        if sort_by:
            df = df.sort_values(sort_by, ascending = ascending)
        fig = px.scatter(df, x=df[categories[0]], y=df[values[0]],
                         size=df[values[1]], color=df[categories[1]],
                         hover_name=hover_name, size_max=20)
        fig.update_traces(dict(showlegend=True))
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def stacked_area_chart(self, time: str, values: list, aggfunc: str='mean', data_labels: bool=True, sort_by: str=None, ascending: bool=True,
                           legend: list=None, unit: str=None, width: int=600, height: int=500, title: str="Stacked Area Chart") -> object:
        """This method will plot a stacked area chart with following arguments provided:

        Args:
            time (str): A time or any relevant column name to be presented on x-axis.
            values (list): A list of numerical column to be ploted against the x-axis.
            aggfunc (str, optional): Aggregation function. Defaults to 'mean'.
            data_labels (bool, optional): Show data labels. Defaults to True.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            legend (list, optional): List of legend names. Defaults to None.
            unit (str, optional): Name of the measurement unit. Defaults to None.
            width (int, optional): Width of the column. Defaults to 600.
            height (int, optional): Height of the column. Defaults to 500.
            title (str, optional): Title of the plot. Defaults to "Stacked Area Chart".

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        if sort_by in values:
            df = self.df.groupby(time, as_index=False)\
            .agg(aggfunc, numeric_only=True)[[time]+values]
            if sort_by:
                df = df.sort_values(sort_by, ascending = ascending)
        elif sort_by is not None:
            df = self.df.groupby(time, as_index=False)\
            .agg(aggfunc, numeric_only=True)[[time]+values+[sort_by]]
            if sort_by:
                df = df.sort_values(sort_by, ascending = ascending)
        else:
            df = self.df.groupby(time, as_index=False)\
            .agg(aggfunc, numeric_only=True)[[time]+values]

        fig = go.Figure()
        if legend is None:
            legend = values
        for i in range(len(values)):
            if data_labels:
                if unit == None:
                    unit = " "
                else:
                    unit = " "+unit
                text = [str(round(i, 0))+unit for i in df[values[i]]]
            else: text = None
            fig.add_trace(go.Scatter(
                x=df[time], y=df[values[i]],
                hoverinfo='x+y',
                mode='lines+markers+text',
                text=text,
                name=legend[i],
                textposition='top left',
                line=dict(width=0.5),
                marker=dict(size=7),
                stackgroup='one'  # define stack group
            ))

        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def scatterplot(self, values: list, size: str=None, color: str=None, hover_name: str=None, width: int=600, height :int=450, title: str='Scatter chart') -> object:
        """This method will plot a scatter plot with the following parameters provided:

        Args:
            values (list): List of the names of numerical columns to be plotted against each other.
            size (str, optional): Name of the column to be presented as the size of the data point. Defaults to None.
            color (str, optional): Name of the column to differentiate data points on color. Defaults to None.
            hover_name (str, optional): Name of the column to be presented as the name on hover. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to 'Scatter chart'.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        fig = px.scatter(self.df, x=values[0], y=values[1],
                         size=size, color=color,
                         hover_name=hover_name)
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)

        return fig

    def facetgrid(self, categories: str, values: str, facet_row: str, facet_col: str, color: str=None, sort_by: str=None, ascending: bool=True,
                  aggfunc: str='mean', barmode: str='stack', width: int=600, height: int=450, title: str='Facetgrid') -> object:        
        """This method will plot the facet grid with the following arguments provided:

        Args:
            categories (str): Name of the categorical column.
            values (str): Name of the numerical column.
            facet_row (_type_): Name of the column to be plotted as a facet row.
            facet_col (_type_): Name of the column to be plotted as a facet column.
            color (str, optional): Name of the column to differentiate data points on color. Defaults to None.
            sort_by (str, optional): Name of the column to sort the data on. Defaults to None.
            ascending (bool, optional): Sorting order. Defaults to True.
            aggfunc (str, optional): Aggregation function. Defaults to 'mean'.
            barmode (str, optional): Chart could be plotted with the stack of group of bars. Defaults to 'stack'.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 450.
            title (str, optional): Title of the plot. Defaults to 'Facetgrid'.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        cols_list = [categories]+[color]+[facet_col]+[facet_row]+[values]
        df = self.df.groupby(cols_list[:-1], as_index=False)\
            .agg(aggfunc, numeric_only=True)[cols_list]
        if sort_by:
            df = df.sort_values(sort_by, ascending = ascending)
        fig = px.bar(df, x=categories, y=values, color=color, barmode=barmode,
                     facet_row=facet_row, facet_col=facet_col,
                     category_orders={facet_row: df[facet_row].tolist(),
                                      facet_col: df[facet_col].tolist()})
        fig.update_layout(autosize=False, width=width,
                    height=height,title=title, title_x=0.5)
        return fig

    def pareto_chart(self, categories: str, values: str, data_labels: bool=True, unit: str=None, unit_position : str="after",
                      y_label: str=None, width: int=600, height: int=500, title: str='Pareto Chart') -> object:       
        """This method will plot a pareto chart with the following arguments provided:

        Args:
            categories (str): Categorical column name.
            values (str): Numerical column name.
            data_labels (bool, optional): Show data labels. Defaults to True.
            unit (str, optional): Unit of measurement to plot with data labels. Defaults to None.
            unit_position (str, optional): Position of the units of data labels, which could be before or after the label values. Defaults to 'after'.
            y_label (str, optional): A custom name for the y-label. Defaults to None.
            width (int, optional): Width of the plot. Defaults to 600.
            height (int, optional): Height of the plot. Defaults to 500.
            title (str, optional): Title of the plot. Defaults to 'Pareto Chart'.

        Returns:
            fig(object): An Object which can be used to save or plot charts in any python application.
        """
        if y_label is None:
            y_label = values
        if unit == None:
            unit = " "
        else:
            unit = " "+unit
        df = self.df.groupby([categories], as_index=False).agg(
            "sum", numeric_only=True)[[categories]+[values]]
        df.sort_values(by=values, ascending=False, inplace=True)
        
        df["cumulative_%"] = 100 * (df[values].cumsum() / df[values].sum())

        if data_labels:
            try:
                trace0_text = [str(int(val))+unit if unit_position == 'after' else unit+' '+str(int(val)) for val in df[values]]
            except ValueError:
                trace0_text = [str(round(float(val),0))+unit if unit_position == 'after' else unit+' '+str(round(float(val),0)) for val in df[values]]

            trace1_text = [str(round(i, 0))+"%" for i in df['cumulative_%']]
        else: 
            trace0_text = None
            trace1_text = None
        trace_0 = go.Bar(
            x=df[categories],
            y=df[values],
            marker=dict(color=df[values], coloraxis="coloraxis"),
            text=trace0_text,
            textposition='auto'
        )


        trace_1 = go.Scatter(
            x=df[categories],
            y=df["cumulative_%"],
            text=trace1_text,
            mode="lines+markers+text",
            textposition="top left"
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(trace_0)

        fig.add_trace(trace_1, secondary_y=True)

        fig.update_layout(
            title=title,
            title_x=0.5,
            autosize=False,
            width=width, height=height,
            yaxis={"title": y_label},
            showlegend=False,
            coloraxis_showscale=False
        )

        return fig
