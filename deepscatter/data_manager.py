import pandas as pd
import numpy as np
import os

# Rircordarsi di specificare il formato del dataframe che deve avere specificando le colonne e il tipo

def load_data() -> pd.DataFrame | Exception:
    """
    Load sample data to use with deepscatter

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sample data for testing deepscatter.
    
    Raises
    ------
    FileNotFoundError
        If the sample data file cannot be found.
    Exception
        For any other unexpected errors.
    """

    try:
        data_dir = os.path.dirname(__file__)
        data_path = os.path.join(data_dir, 'data', 'sample.csv')
        return pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Sample data file not found: {data_path}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading the data: {str(e)}") from e

def mark_anomalies_in_timeseries(df:pd.DataFrame, start_date:str, end_date:str)->pd.DataFrame:
    """
    Transform the data by adding the anomalies information

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of the time series
    start_date: str
        Shift anomaly starting timestamp
    end_date: str
        Shift anomaly ending timestamp
    
    Returns
    -------
    pd.DataFrame
        Reshaped original data

    Raises
    ------
    ValueError
        If the 'timestamp' column is missing in the DataFrame.
    """

    if 'timestamp' not in df.columns:
        raise ValueError("The DataFrame must contain a 'timestamp' column.")
    
    # Adding a new field to the DataFrame to keep track of the anomalies
    df['anomaly'] = 0

    # Setting the anomaly field to 1 (i.e., anomaly) for the interval between the start and end timestamps
    df.loc[(df['timestamp'] > start_date) & (df['timestamp'] < end_date), 'anomaly'] = 1

    return df

def train_test_split(df:pd.DataFrame, normality_threshold_index:int, t:int=20) -> np.array:

    """
    Split the timeserie into train (i.e., normal sample) and test set based on the normality_threshold_index that represents where to split the sequence

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of the timeserie
    normality_threshold_index: int
        Value where to split the timeserie into train and test
    t: int
        Window dimension to aggregate the values
    
    Returns
    -------
    train: np.array
        Train set
    test: np.array
        Test set

    Raises
    ------
    ValueError
        If the 'timestamp' column is missing in the DataFrame.
    IndexError
        If the normality_threshold_index is not within the bounds of the DataFrame
    """

    df.reset_index(inplace=True)

    if 'value' not in df.columns:
        raise ValueError("The DataFrame must contain a 'timestamp' column.")
    
    if normality_threshold_index < 0 or normality_threshold_index > len(df):
        raise IndexError("normality_threshold_index is out of bounds.")

    #Splitting the dataframe column "value" into train and test based on the normality_threshold_index
    train_ = np.array(df[0:1700]["value"])
    test_ = np.array(df[1700:3000]["value"])

    #Computing the padding whether needed to added based on the value of t
    train_padding = t - np.mod(len(train_), t)
    test_padding = t - np.mod(len(test_), t)

    #Stacking the padding array to the train and test sets
    train_ = np.hstack((train_, np.tile(train_[-1], train_padding)))
    test_ = np.hstack((test_, np.tile(train_[-1], test_padding)))

    #Computing the new shape
    train_newshape = (len(train_)//t, t)
    test_newshape = (len(test_)//t, t)

    train = np.reshape(train_, train_newshape)
    test = np.reshape(test_, test_newshape)

    return train, test






