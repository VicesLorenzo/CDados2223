from pandas import DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from global_var import * 
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, show



def Standart_Scaler_zscore(file,data,numeric_vars,df_nr,df_sb,df_bool):
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
    #norm_data_zscore.to_csv(f'../../data/week2/{file}_scaled_zscore.csv', index=False)
    #print(norm_data_zscore.describe())


def MinMaxScaler_(file,data,numeric_vars,df_nr,df_sb,df_bool):
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_minmax.to_csv(f'../../data/week2/{file}_scaled_minmax.csv', index=False)
    print(norm_data_minmax.describe())

def scalling(file,data):
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars] #table with numeric vars
    df_sb = data[symbolic_vars] #table with symbolic vars
    df_bool = data[boolean_vars] #table with bool vars

    #Standart_Scaler_zscore(file,data,numeric_vars,df_nr,df_sb,df_bool)
    MinMaxScaler_(file,data,numeric_vars,df_nr,df_sb,df_bool)



def main():
    data_health,data_climate = gbVar()
    
    #scalling(filename_h,data_health)
    scalling(filename_c,data_climate)


if __name__=="__main__":
    main()