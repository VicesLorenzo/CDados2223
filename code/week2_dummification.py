from globals import class_climate_encoded_df, class_health_encoded_df
from pandas import DataFrame, concat
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder

def dummify(df, vars_to_dummify):
    if vars_to_dummify == []:
        print("No dummification required")
        return df
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)
    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

df = dummify(class_climate_encoded_df, get_variable_types(class_climate_encoded_df)["Symbolic"])
df = dummify(class_health_encoded_df, get_variable_types(class_health_encoded_df)["Symbolic"])
