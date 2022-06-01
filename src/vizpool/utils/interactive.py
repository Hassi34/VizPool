import plotly.graph_objects as go
import plotly.express as px

def barchart(df, values, categories, title = 'Bar Chart', orientation='h', texposition = "auto", text_color ='white', width=600, height=500 ):
    """It will receive the following parameters and return the "fig" object to plot a Barchart

    Args:
        df (object): A pandas dataframe object
        values (int, float): Numerical data to plot against the categories
        categories (object): An object list containing the categories as elements
        title (str): The title of the chart. Defaults to 'Bar Chart'.
        orientation (str): To plot the chart horizontly or vertically. Defaults to 'h'.
        texposition (str): Position of the text on chart. Defaults to "auto".
        text_color (str): Text color. Defaults to 'white'.
        width (int): The width of the visualization. Defaults to 600.
        height (int, optional): The height of the visualization. Defaults to 500.

    Returns:
        object: it returns figure object, calling .show() on the object will plot the chart
    """
    if orientation == 'h':
        data_labels = values
    else:
        data_labels = categories
    fig = go.Figure(go.Bar(
                x=df[values],
                y=df[categories],
                orientation= orientation,
                text=[int(val) for val in df[data_labels]],textposition=texposition,textfont=dict(color=text_color),))

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.update_layout(autosize=False,width=600,height=500,title=title)
    return fig

def heatmap(df, index, columns, values, aggfunc, x_label = None, y_label = None, color_label = None, title = None):
    df_pivot = df[[index, columns ,values]].pivot_table(index =index, columns = columns, aggfunc = aggfunc, fill_value=0)
    fig = px.imshow(df_pivot.values,
            labels=dict(x=x_label, y=y_label, color=color_label),
            x=[x[1] for x in df_pivot.columns],
            y=df_pivot.index.tolist(), text_auto=True
            )
    fig.update_traces(dict(showscale=False, 
                       coloraxis=None, 
                       colorscale='bluered'), selector={'type':'heatmap'})
    fig.update_xaxes(side="top", title = title)
    return fig 
def stack_or_group_chart(df,cat_var, num_var,barmode="stack", orientation="h", texposition = 'inside', text_color = 'white', title=None):
    if len(num_var) == 4:
        fig = go.Figure(data=[
            go.Bar(name=num_var[0], y=df[cat_var], x=df[num_var[0]], orientation=orientation,
            text=[int(val) for val in df[num_var[0]]],textposition=texposition,textfont=dict(color=text_color),),
            go.Bar(name=num_var[1], y=df[cat_var], x=df[num_var[1]], orientation=orientation,
            text=[int(val) for val in df[num_var[1]]],textposition=texposition,textfont=dict(color=text_color),),
            go.Bar(name=num_var[2], y=df[cat_var], x=df[num_var[2]], orientation=orientation,
            text=[int(val) for val in df[num_var[2]]],textposition=texposition,textfont=dict(color=text_color),),
            go.Bar(name=num_var[3], y=df[cat_var], x=df[num_var[3]], orientation=orientation,
            text=[int(val) for val in df[num_var[3]]],textposition=texposition,textfont=dict(color=text_color),)
            ])
    elif len(num_var) == 3:
        fig = go.Figure(data=[
            go.Bar(name=num_var[0], y=df[cat_var], x=df[num_var[0]], orientation=orientation,
            text=[int(val) for val in df[num_var[0]]],textposition=texposition,textfont=dict(color=text_color),),
            go.Bar(name=num_var[1], y=df[cat_var], x=df[num_var[1]], orientation=orientation,
            text=[int(val) for val in df[num_var[1]]],textposition=texposition,textfont=dict(color=text_color),),
            go.Bar(name=num_var[2], y=df[cat_var], x=df[num_var[2]], orientation=orientation,
            text=[int(val) for val in df[num_var[2]]],textposition=texposition,textfont=dict(color=text_color),)
            ])
    elif len(num_var) == 2:
        fig = go.Figure(data=[
        go.Bar(name=num_var[0], y=df[cat_var], x=df[num_var[0]], orientation=orientation,
        text=[int(val) for val in df[num_var[0]]],textposition=texposition,textfont=dict(color=text_color),),
        go.Bar(name=num_var[1], y=df[cat_var], x=df[num_var[1]], orientation=orientation,
        text=[int(val) for val in df[num_var[1]]],textposition=texposition,textfont=dict(color=text_color),)
        ])
    fig.update_layout(barmode=barmode)
    fig.update_layout(autosize=False,width=600,height=500,title=title)
    return fig


def pie_chart(df, values, categories, title='Pie Chart'):
    fig = px.pie(data_frame=df, values=values,names=categories,
           color=categories,                      #differentiate markers (discrete) by color
        #color_discrete_sequence=["red","green","blue"],     #set marker colors
        # color_discrete_map={"WA":"yellow","CA":"red","NY":"black","FL":"brown"},
        #hover_name='Nature',              #values appear in bold in the hover tooltip
        #hover_data=['Nature', 'Category'],            #values appear as extra data in the hover tooltip
        # custom_data=['total'],       #map the labels
        title=title,     #figure title
        template='presentation',            #'ggplot2', 'seaborn', 'simple_white', 'plotly',
                                        #'plotly_white', 'plotly_dark', 'presentation',
                                        #'xgridoff', 'ygridoff', 'gridon', 'none'
        width = 600,                          #figure width in pixels
        height=500,                         #figure height in pixels
        hole=0.5,                           #represents the hole in middle of pie
        )
    return fig
def area_chart(df, categories, values, x_label, y_label, title, unit=None):
    if unit == None:
        unit = " "
    else:
        unit = " "+unit
    fig = px.area(
    df,
    x= df[categories],
    y= df[values],
    labels={
                    "x": x_label,
                    "charges":y_label},
    text = [str(int(val))+unit for val in df[values]],title=title
    )
    fig.update_traces(mode='markers+lines+text', textfont_size=12, textposition="top left")
    fig.update_layout(autosize=False,width=600,height=450,)
    return fig
def line_bar(df, x, y1, y2 ,legends,title, texposition = "auto", text_color ="white" ,round_decimal = 0 ):
    if round_decimal > 0:
        line_data_labels = [round(val, round_decimal) for val in y1]
        bar_data_labels = [round(val, round_decimal) for val in y2]
    elif round_decimal == 0:
        line_data_labels = [int(val) for val in y1]
        bar_data_labels = [int(val) for val in y2]
    else:
        print("You cannot have a negative number for decimal")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name = legends[0],x=x,y=y1,
        text=line_data_labels, textposition='top right',
        mode='lines+markers+text',marker=dict(color='#5D69B1', size=7)
        ))
    fig.add_trace(
        go.Bar(name = legends[1],x=x,y=y2,
        text=bar_data_labels,textposition=texposition,textfont=dict(color=text_color),)
        )
    fig.update_layout(autosize=False,width=600,height=500,title=title, 
                      legend=dict(xanchor="auto")) # yanchor="top",y=0.99 ,x=0.01
    return fig
def plot_histogram(df, x , y , color, marginal, hover_data, title, width, height):
    fig = px.histogram(df, x=x, y=y, color=color, 
    marginal=marginal,histfunc='count', hover_data=hover_data)
    fig.update_layout(autosize=False,width=width,height=height,title = title)
    return fig

def combined_corr(df, x, y, color, hover_name, size, title ):
    fig = px.scatter(df, x=x, y=y, color=color,
                            marginal_x = "box" , marginal_y = 'violin', hover_name = hover_name,
                            size=size, title = title )
    return fig

def multivar_bubble(df, col1, col2, val1, val2, aggfunc = "mean",hover_name = None , x_label = None, y_label = None, color_label = None, title = None):
  df_t = df.groupby([col1, col2]).agg(aggfunc)[[val1, val2]]
  fig = px.scatter(df, x=col1, y=val1,
            size=val2, color=col2,
                  hover_name=hover_name, size_max=20)
  fig.update_traces(dict(showlegend=True))
  fig.update_layout(autosize=False,width=600,height=450,)
  fig.update_xaxes(side="top", title = title)
  return fig 

def stacked_area_chart(df, x, y1, y2, legend, title):
    
    x= df[x].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df[y1].tolist(),
        hoverinfo='x+y',
        mode='lines',
        name = legend[0],
        line=dict(width=0.5, color='rgb(131, 90, 241)'),
        stackgroup='one' # define stack group
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df[y2].tolist(),
        hoverinfo='x+y',
        name = legend[1],
        mode='lines',
        line=dict(width=4.4, color='rgb(111, 231, 219)'),
        #stackgroup='one'
    ))
    # fig.add_trace(go.Scatter(
    #     x=x, y=y3,
    #     hoverinfo='x+y',
    #     name = None,
    #     mode='lines',
    #     line=dict(width=0.5, color='rgb(184, 247, 212)'),
    #     stackgroup='one'
    # ))
    fig.update_layout(autosize=False,width=600,height=500,title=title, 
                    legend=dict(xanchor="auto"))
    return fig
def bubble_chart(df, x, y, size , color, hover_name, title):
    fig = px.scatter(df, x=x, y=y,
	        size=size, color=color,
            hover_name=hover_name)
    fig.update_layout(autosize=False,width=600,height=450,title = title)

    return fig
def facetgrid(df, cat_cols, value , barmode = 'stack', title = 'Facetgrid'):
    fig = px.bar(df, x=cat_cols[0], y=value, color=cat_cols[1], barmode=barmode,
             facet_row=cat_cols[2], facet_col=cat_cols[3],
             category_orders={cat_cols[2]: df[cat_cols[2]].tolist(),
                              cat_cols[3]: df[cat_cols[3]].tolist()})
    fig.update_layout(autosize=False,width=600,height=450,title = title)
    return fig