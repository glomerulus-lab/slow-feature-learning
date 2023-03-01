import json
import pandas as pd 

"""
Takes a json file that is formated in the programs output form.
Returns the list of hyperparameters used to run the model and a dataframe of the recorded metrics.
"""
def json_to_dataframe(json_file):
    f = open(json_file)
    data = json.load(f)
    hyper_parameters = data.pop('Hyper Parameters')
    df = pd.DataFrame.from_dict(data)
    return hyper_parameters, df

"""
Takes an array of json files (generated from the model) and a specified column 
Returns a dataframe of the coulmns from the datasets.
"""
def merge_on_column(json_file_array, column):
    merged_df = pd.DataFrame()
    for json_file in json_file_array:
        hyper_paramters, df = json_to_dataframe(json_file)
        column_name = str(hyper_paramters[-2]) + ', ' + str(hyper_paramters[-1]) + ' ' + column # Coulumn Title
        merged_df[column_name] = df[column]
    return merged_df

def merge_on_metric(digits, metric):
    dfs = [None] * 16
    for j in range(0, 4):
        file_name = "data/" + "r" + digits + "(1)" * j + ".json"
        dfs[j] = file_name
    for i in range(0,12):
        file_name = "data/" +"s" + digits + "(1)" * i + ".json"
        dfs[i + 4] = file_name
    
    #for file_name in dfs:
        #dfs[file_name] = merge_on_column(file_name, metric)
    return merge_on_column(dfs, metric)