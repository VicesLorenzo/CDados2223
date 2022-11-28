from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig,show, figure, savefig, show, title
from ds_charts import get_variable_types, HEIGHT
from seaborn import heatmap
import matplotlib

matplotlib.use('Agg')


register_matplotlib_converters()
filename_health = '../../datasets/classification_health.csv'
filename_climate = '../../datasets/classification_climate.csv'
data_health = read_csv(filename_health, index_col='encounter_id',parse_dates=True, infer_datetime_format=True)
data_climate = read_csv(filename_climate, index_col='date',parse_dates=True, infer_datetime_format=True)

def numeric(data):
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(f'../../images/week1/sparsity_study_numeric.png')
    #show()

def symbolic(data):

    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(f'../../images/week1/sparsity_study_symbolic.png')
    # show()


def correlaction(data):
    corr_mtx = abs(data.corr())
    print(corr_mtx)
    fig = figure(figsize=[12, 12])

    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(f'../../images/week1/correlation_analysis.png')
    # show()

def main():

    #numeric(data_health)
    #symbolic(data_health)
    #correlaction(data_health)
    
    numeric(data_climate)
    symbolic(data_climate)
    correlaction(data_climate)

if __name__=="__main__":
    main()
