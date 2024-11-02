import numpy as np;
import pandas as pd;
import constants;
from setup import decodeLabel;

def printConfMtx(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred = decodeLabel(y_pred);
    y_pred = np.resize(y_pred, y_true.shape);
    print(f"Decoded Labels in Prediction array: {np.unique(y_pred)}");

    pandas_y_actual = pd.Series(y_true, name='Actual');
    pandas_y_pred = pd.Series(y_pred, name='Predicted');

    confMtx:pd.DataFrame = pd.crosstab(
        pandas_y_actual, 
        pandas_y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'], 
        margins=True);

    class_names = constants.LABELS;

    new_row_items = new_col_items = class_names + ['All'];

    # Convert to DataFrame with class names
    #confMtx_df = pd.DataFrame(confMtx, index=class_names, columns=class_names);
    confMtx = confMtx.reindex(index=new_row_items, columns=new_col_items);
    confMtx.index.name = 'Actual';
    confMtx.columns.name = 'Predicted';

    confMtx.fillna(0, inplace=True);
    
    return confMtx;