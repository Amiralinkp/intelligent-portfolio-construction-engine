import numpy as np

def calculate_percentile(historical_values, current_value):

    historical_values = historical_values.dropna()

    count = 0
    for val in historical_values:
        if val<= current_value:
            count +=1
    if len(historical_values) == 0:
        return np.nan
    
    total = len(historical_values)
    percentage = count/total

    return percentage