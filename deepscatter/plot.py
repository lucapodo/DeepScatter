import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots

def plot_shit_anomaly(df: pd.DataFrame) -> None:

    """
    Plot the original data along with the provided anomaly label

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with time serie and anomaly labels
    
    """

    # Define a new plot with 'timestamp' as the x-axis and 'value' as the y-axis.
    fig = go.Figure()

    # Add the non-anomaly points with green color
    fig.add_trace(go.Scatter(
        x=df[df['anomaly'] == 0]['timestamp'],
        y=df[df['anomaly'] == 0]['value'],
        mode='markers',
        marker=dict(color='green', size=4),
        name='Normal'  # Legend label for good data points
    ))

    # Add the anomaly points with red color
    fig.add_trace(go.Scatter(
        x=df[df['anomaly'] == 1]['timestamp'],
        y=df[df['anomaly'] == 1]['value'],
        mode='markers',
        marker=dict(color='red', size=4),
        name='Shift anomaly'  # Legend label for anomaly data points
    ))

    # Add a horizontal line representing the mean value
    mean_value = df['value'].mean()
    fig.add_shape(
        type='line',
        x0=df['timestamp'].min(),
        y0=mean_value,
        x1=df['timestamp'].max(),
        y1=mean_value,
        line=dict(color='black', dash='dash'),
        name='Mean Value'  # Legend label for the mean line
    )

    # Set the title and axis labels for the plot, and show the legend
    fig.update_layout(
        title=dict(text='Timeserie', x=0.5),
        xaxis_title='Timestamp',
        yaxis_title='Values',
        showlegend=True,
    )

    # Display the plot.
    fig.show()

def plotD(train_scores: np.ndarray, 
                            test_scores: np.ndarray, 
                            df: pd.DataFrame, 
                            l: int = 5, 
                            t: int = 20) -> None:
    
    """
    Creates a plot comparing train and test scores with anomaly detection values.

    Parameters
    ----------
    train_scores : np.ndarray
        Array of training scores, must be a 1D numpy array.
    test_scores : np.ndarray
        Array of testing scores, must be a 1D numpy array.
    df : pd.DataFrame
        DataFrame containing anomaly information with columns 'anomaly' and 'value'.
    t : int, optional
        Number of repetitions for each score in train_scores and test_scores (default is 20).

    Raises
    ------
    ValueError
        If `train_scores` or `test_scores` are not 1D numpy arrays.
    KeyError
        If `df` does not contain the required columns 'anomaly' and 'value'.
    """

    if not isinstance(train_scores, np.ndarray) or train_scores.ndim != 1:
        raise ValueError("train_scores must be a 1D numpy array.")
    if not isinstance(test_scores, np.ndarray) or test_scores.ndim != 1:
        raise ValueError("test_scores must be a 1D numpy array.")
    if 'anomaly' not in df.columns or 'value' not in df.columns:
        raise KeyError("DataFrame must contain 'anomaly' and 'value' columns.")

    #Repeating the element t times to recreate the lenght original sequence on point
    train_scores = np.repeat(train_scores, t)
    test_scores = np.repeat(test_scores, t)

    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)

    y = list(train_scores) + list(test_scores)
    x = list(range(len(y)))

    c1 = ['green' if a == 0 else 'red' for a in df['anomaly']]
    c2 = ['orange' for _ in range(len(train_scores))] + c1[len(train_scores):-1]

    y += [1, -1]
    scaler = MinMaxScaler()
    y = scaler.fit_transform(np.asarray(y).reshape(-1, 1))
    y = list(y)
    y = y[0:-2]

    mean = np.asarray(y[0:len(train_scores)]).mean()
    std = np.std(y[0:len(train_scores)])
    min_val = mean - 2 * std
    max_val = mean + 2 * std
    y = [data[0] for data in y]

    # Initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("DeepScatter", "Vanilla scatter"), shared_xaxes=True, shared_yaxes=False
    )

    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=c2)), row=1, col=1)
    fig.add_shape(type="line", x0=0, x1=len(y)-1, y0=mean, y1=mean, line=dict(color='gray', width=1, dash='dash'), name="Mean", row=1, col=1)
    fig.add_shape(type="line", x0=0, x1=len(y)-1, y0=min_val, y1=min_val, line=dict(color='blue', width=1, dash='dash'), name="Min", row=1, col=1)
    fig.add_shape(type="line", x0=0, x1=len(y)-1, y0=max_val, y1=max_val, line=dict(color='blue', width=1, dash='dash'), name="Max", row=1, col=1)
    fig.update_xaxes(title="Timestep", row=1, col=1)
    fig.update_yaxes(title="Similarity score", row=1, col=1)


    # Add an empty subplot to the second column
    yy = df['value']
    fig.add_trace(go.Scatter(x=x, y=yy, mode='markers', marker=dict(color=c1), showlegend= True), row=2, col=1)
    fig.update_xaxes(title="Timestep", row=2, col=1)
    fig.update_yaxes(title="Values", row=2, col=1,side='left')


    # Add the second scatter plot to the second column
    fig.update_layout(title="DeepScatter vs. Vanilla Scatter", showlegend=False, xaxis2=dict(showgrid=True, zeroline=True, showticklabels=True), yaxis2=dict(showgrid=True, zeroline=True, showticklabels=True))
    fig.show()

