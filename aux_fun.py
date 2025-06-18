import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html, Input, Output, State
from plotly.subplots import make_subplots
# Función para graficar variables númericas vs variable objetivo de tipo binario
def hist_meanplot_numeric(df: pd.DataFrame,
    target: str,
    variable: str,
    quantiles: tuple = (0.05, 0.95),
    n_bins: int = 18,
    color = '#cc0033',
    figsize: tuple = (15, 10)):
    # Calculate the percentiles of the feature column
    q_5 = df[variable].quantile(quantiles[0])
    q_95 = df[variable].quantile(quantiles[1])
    # Filter df by quantiles
    condition = (df[variable]>=q_5)&(df[variable] <= q_95)
    # Create equal-width bins for the variable column
    bins = np.linspace(df[condition][variable].min(),df[condition][variable].max(),7)
    # Group the data by bins and calculate the mean of the target column for each bin
    df_grouped = df.loc[condition,[variable,target]].groupby(pd.cut(df[variable],bins=bins), observed=False).mean()
    # Calculate the midpoint of each bin
    bin_midpoints= pd.IntervalIndex(df_grouped.index).mid

    # Plot the histogram with outlier removal
    fig, ax1 = plt.subplots()
    sns.histplot(data=df[condition], x=variable, bins=bins, kde=False,stat='percent', ax=ax1,color=color)

    # Plot the line plot with target mean values for each bin
    ax2 = ax1.twinx()
    sns.lineplot(data=df_grouped, x=bin_midpoints, y=target,color="#000000",markers=True, marker='o',markersize=4,ax=ax2)
    # Set the x-axis label
    if quantiles == (0, 1):
        ax1.set_xlabel(f'{variable}')
    else:
        ax1.set_xlabel(f'{variable} (outliers removed)')
    ax2.set_ylabel(f'Mean {target} by {variable}')
    # Show the plot
    plt.figure(figsize=figsize)
    plt.show()

def histplot_mean_target_categories(
    df: pd.DataFrame,
    target: str,
    variable: str,
    max_categories: int = 10,
    order: bool = False,
    color = '#cc0033',
    figsize: tuple = (15, 10)):
    
    df_1 = df[[variable,target]].copy()
    # transform the variable column into a categorical column
    #df_1[variable] = df_1[variable].astype('category')

    # if the number of categories is greater than max_categories, group the non-top categories into 'Other'
    if df_1[variable].nunique() > max_categories:
        top_categories = df_1[variable].value_counts().index[:max_categories]
        df_1.loc[~df_1[variable].isin(top_categories), variable] = 'Other'
    
    # Group the data by bins and calculate the mean of the target column for each bin
    df_grouped = df_1.loc[:, [variable,target]].groupby([ variable]).mean()
    df_grouped.reset_index(inplace=True)
    
    #order
    if order:
        # natural order
        order = None
    else:
        order = df_1[variable].value_counts().index
    
    # Plot the histogram 
    fig, ax1 = plt.subplots()
    sns.countplot(data=df_1, x=variable, stat='percent', ax=ax1,color=color, order = order)
    ax1.tick_params(axis='x', rotation=45)
    # Plot the line plot with target mean values for each bin
    ax2 = ax1.twinx()
    sns.lineplot(data=df_grouped, x=variable, y=target, color="#000000",markers=True, marker='o',markersize=4,ax=ax2)
    
    # Set the x-axis label
    ax1.set_xlabel(f'{variable}')
    ax2.set_ylabel(f'Mean {target} by {variable}')
    # Show the plot
    plt.figure(figsize=figsize)
    plt.show()

def detect_outliers(
        df:pd.DataFrame,
        num_cols:list,
        ):
    variable=[]
    contador = []
    sini =[]
    outliers_threshold = []
    for col in num_cols:
        media = np.mean(df[col])
        sigma = np.std(df[col])
        outliers = df[(df[col] < (media - 3 * sigma)) | (df[col] > (media + 3 * sigma))]
        
        variable.append(col)
        contador.append(len(outliers))
        outliers_threshold.append(round(media+3*sigma,0))
        #sini.append((outliers['siniestro']==1).sum())
        data = {
            'Variable':variable ,
            'Num. outliers':contador,
            'Outlier Threshold': outliers_threshold
            #'Num. Siniestros':sini
        }

    resumen = pd.DataFrame(data)
    resumen['% Outliers']=round((resumen['Num. outliers']/len(df))*100,2)
    #resumen['% Siniestro']=round((resumen['Num. Siniestros']/(df['siniestro']==1).sum())*100,2)
    resumen.sort_values(by=['% Outliers'])
    return resumen

# Create a dash function for plotting numeric columns
def graph_hist_numeric(
        df:pd.DataFrame,
        num_cols:list
):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Distribution of Numeric Columns'),
        dcc.Dropdown(id='dropdown',options=num_cols,value='coste_siniestro'),
        dcc.Graph(id='histogram')
    ])

    @app.callback(Output('histogram','figure'),
                Input('dropdown','value'))

    def histograma(col):
        fig = px.histogram(df,x=col)
        return fig

    app.run(debug=True)

def tabla_siniestralidad_num(
        df:pd.DataFrame,
        col: str,
        range: int):
    num_grupos = np.linspace(df[col].min(),df[col].max(),range)
    tabla= df.groupby(by=pd.cut(df[col],num_grupos),observed=True)[['coste_siniestro','NSin','exposicion','prima']].sum()\
                                                                                        .assign(severidad=lambda x: x['coste_siniestro']/x['NSin'])\
                                                                                        .assign(Frecuencia = lambda x : x['NSin']/x['exposicion'])\
                                                                                        .assign(PrimaPura = lambda x : x['coste_siniestro']/x['exposicion'])\
                                                                                        .assign(RC= lambda x : x['coste_siniestro']/x['prima'])
    return tabla

def tabla_siniestralidad_cat(
        df:pd.DataFrame,
        values:list,
        category:str):
        
    tabla =pd.pivot_table(
        df,
        values=values,
        index=category,
        aggfunc='sum')\
            .assign(severidad=lambda x: x['coste_siniestro']/x['NSin'])\
            .assign(Frecuencia = lambda x : x['NSin']/x['exposicion'])\
            .assign(PrimaPura = lambda x : x['coste_siniestro']/x['exposicion'])\
            .assign(RC= lambda x : x['coste_siniestro']/x['prima'])
    return tabla

def lossratio_graph(
        df:pd.DataFrame,
        category:str):
    tabla_resumen=tabla_siniestralidad_cat(df,['coste_siniestro','prima','NSin','exposicion'],category)
    fig =make_subplots(specs=[[{'secondary_y':True}]])
    # primary axis
    fig.add_trace(go.Bar(x=tabla_resumen.index,y=tabla_resumen['PrimaPura'],name='Pure Premium'))
    # secondary axis
    fig.add_trace(go.Scatter(x=tabla_resumen.index,y=tabla_resumen['RC'],name='Loss Ratio'),secondary_y=True)
    # horizontal line
    fig.add_hline(y=1,yref='y2',line_width=3,line_dash='dash',line_color='red')
    fig.update_layout(title_text='Economic Situation Summary')
    # Set x-axis title
    fig.update_xaxes(title_text=tabla_resumen.index.name.upper())
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Loss Ratio</b> value", secondary_y=True)
    fig.update_yaxes(title_text="Mean <b>Premium</b>", secondary_y=False)
    return fig

def accident_summary(df:pd.DataFrame,category:str):
    tabla_resumen=tabla_siniestralidad_cat(df,['coste_siniestro','prima','NSin','exposicion'],category)
    # Round values and format numbers
    tabla_formatted = tabla_resumen.round(2).map(lambda x: f"{x:,.2f}")

    # Define colors for RC column
    rc_values = tabla_resumen['RC']
    colors = []
    for val in rc_values:
        if val > 1:
            colors.append('red')
        elif val >= 0.7:
            colors.append("orange")
        else:
            colors.append("green")

    # Create the Plotly table
    header_values = [category] + tabla_formatted.columns.tolist()
    cell_values = [tabla_formatted.index.tolist()] + [tabla_formatted[col].tolist() for col in tabla_formatted.columns]

    # Define fill colors for cells
    fill_colors = [['white'] * len(tabla_formatted)] * len(header_values)
    rc_index = header_values.index('RC')
    fill_colors[rc_index] = colors

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    fill_color='lightgrey',
                    align='center'),
        cells=dict(values=cell_values,
                fill_color=fill_colors,
                align='center'))
    ])

    fig.update_layout(title='Economic Situation Summary',width=1000, height=400)
    return fig


def plot_obs_pred(
    df,
    feature,
    weight,
    observed,
    predicted,
    y_label=None,
    title=None,
    ax=None,
    fill_legend=False,
):
    """Plot observed and predicted - aggregated per feature level.

    Parameters
    ----------
    df : DataFrame
        input data
    feature: str
        a column name of df for the feature to be plotted
    weight : str
        column name of df with the values of weights or exposure
    observed : str
        a column name of df with the observed target
    predicted : DataFrame
        a dataframe, with the same index as df, with the predicted target
    fill_legend : bool, default=False
        whether to show fill_between legend
    """
    # aggregate observed and predicted variables by feature level
    df_ = df.loc[:, [feature, weight]].copy()
    df_["observed"] = df[observed] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["observed", "predicted"]].plot(style=".", ax=ax)
    y_max = df_.loc[:, ["observed", "predicted"]].values.max() * 0.8
    p2 = ax.fill_between(
        df_.index,
        0,
        y_max * df_[weight] / df_[weight].values.max(),
        color="g",
        alpha=0.1,
    )
    if fill_legend:
        ax.legend([p2], ["{} distribution".format(feature)])
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D² explained", None),  # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # Score the model consisting of the product of frequency and
                # severity models.
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append({"subset": subset_label, "metric": score_label, "score": score})

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )
    return res
