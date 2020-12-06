# -*- coding: utf-8 -*-
"""meritvarde-prediktion: prediktionsmodeller för meritvärde i åk 9

Bygger på ett dataset med vårterminens betyg från åk 6 - 9 för elever
födda 1996 - 2008 i Hultsfreds kommun. Data är mer komplett i de senare
årskurserna jämfört med de tidiga.

Prediktionen bygger på att använda alumners meritvärde i åk 9 för att
med olika matematiska modeller göra inferens med ofullständig data.
De modeller som visade bäst tillförlitlighet är linjär regression
och elastiskt nät. Bayesiansk inferens används som komplement.

Inspiration och logik är till stor del hämtad från:
https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Project.ipynb
"""

import mariadb
import sys
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn import linear_model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (9, 9)
import seaborn as sns
from sklearn.utils import shuffle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

import pymc3 as pm

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

model_filename = 'meritvarde-prediktion.pickle'

def evaluate_predictions(predictions, real):
    mae = np.mean(abs(predictions - real))
    rmse = np.sqrt(np.mean((predictions - real) ** 2))

    return mae, rmse


def evaluate_models_iterate(X, y, repeat=5):
    model_name_list = ['Linear regression', 'Gradient boosted', 'Elastic net', 'Random forest', 'Extra trees', 'SVR']

    model1 = LinearRegression()
    model2 = GradientBoostingRegressor(n_estimators=20)
    model3 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model4 = RandomForestRegressor(n_estimators=50)
    model5 = ExtraTreesRegressor(n_estimators=50)
    model6 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')

    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)
    best_mae = 50
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.1)

    print("Repeating", repeat, "times with", len(model_name_list), "models to find best performer..")

    for x in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        print("Evaluating models with different splits, round", x, "/", repeat, "..")

        for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Metrics
            mae, rmse = evaluate_predictions(predictions, y_test)

            model_name = model_name_list[i]
            results.loc[model_name, :] = [mae, rmse]
            mae_value = results['mae'].sort_values().values[0]

            if mae_value < best_mae:
                best_mae = mae_value; best_model = model_name
                xtr, xte, ytr, yte = X_train, X_test, y_train, y_test
                print("New best! MAE:", mae_value, "Model:", model_name, "Saving to pickle file..")
                pickle.dump(model, open(model_filename, 'wb'))

    baseline = np.median(y_train)
    baseline_mae, baseline_rmse = evaluate_predictions(baseline, y_test)
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    print("Lowest MAE:", best_mae, "achieved with", best_model)

    return xtr, xte, ytr, yte, best_model, best_mae


def evaluate_models(X_train, X_test, y_train, y_test):
    model_name_list = ['Linear regression', 'Gradient boosted', 'Elastic net', 'Random forest', 'Extra trees', 'SVR']

    model1 = LinearRegression()
    model2 = GradientBoostingRegressor(n_estimators=20)
    model3 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model4 = RandomForestRegressor(n_estimators=50)
    model5 = ExtraTreesRegressor(n_estimators=50)
    model6 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')

    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)

    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae, rmse = evaluate_predictions(predictions, y_test)

        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]

    baseline = np.median(y_train)
    baseline_mae, baseline_rmse = evaluate_predictions(baseline, y_test)
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    return results


def compare_models(results):
    fig, ax = plt.subplots()
    x = np.arange(len(results.index))
    width = 0.3
    ax.bar(x - width/2, results['mae'].sort_values(ascending=True), width, color='b', label='Mean Absolute Error')
    ax.bar(x + width/2, results['rmse'], width, color='g', label='Root Mean Squared Error')
    ax.set_title("MAE and RMSE evaulation of models\nLower = better")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE and RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(results.sort_values('mae', ascending=True).index)
    ax.legend()
    fig.tight_layout()
    plt.savefig("mae-rmse.svg", format='svg')
    plt.show()


def get_formulas(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    ols_formula = 'Meritvarde = %0.2f +' % lr.intercept_
    for i, col in enumerate(X_train):
        ols_formula += ' %0.2f * %s +' % (lr.coef_[i], col)
    #print(' '.join(ols_formula.split(' ')[:-1]))
    bayesian_formula = 'meritvarde ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[0:]])
    #print(bayesian_formula)
    return ols_formula, bayesian_formula


def bayesian_model(bayesian_formula, dataframe):
    with pm.Model() as normal_model:
        # The prior for the model parameters will be a normal distribution
        family = pm.glm.families.Normal()

        # Creating the model requires a formula and data (and optionally a family)
        pm.GLM.from_formula(bayesian_formula, data=dataframe, family=family)

        # Perform Markov Chain Monte Carlo sampling
        normal_trace = pm.sample(draws=2000, chains=2, tune=2000)

        """
        for variable in normal_trace.varnames:
            print('Variable: {:15} Mean weight in model: {:.4f}'.format(variable, np.mean(normal_trace[variable])))
        """

        return normal_trace, pm


# Examines the effect of changing a single variable
# Takes in the name of the variable, the trace, and the data
def model_effect(query_var, trace, X, filename='model_effect.svg'):
    # Variables that do not change
    steady_vars = list(X.columns)
    steady_vars.remove(query_var)

    # Linear Model that estimates a grade based on the value of the query variable
    # and one sample from the trace
    def lm(value, sample):
        # Prediction is the estimate given a value of the query variable
        prediction = sample['Intercept'] + sample[query_var] * value

        # Each non-query variable is assumed to be at the median value
        for var in steady_vars:
            # Multiply the weight by the median value of the variable
            prediction += sample[var] * X[var].median()

        return prediction

    # Find the minimum and maximum values for the range of the query var
    var_min = X[query_var].min()
    var_max = X[query_var].max()

    # Plot the estimated grade versus the range of query variable
    pm.plot_posterior_predictive_glm(trace, eval=np.linspace(var_min, var_max, 100),
                                     lm=lm, samples=100, color='blue',
                                     alpha=0.4, lw=2)

    # Plot formatting
    plt.xlabel('%s' % query_var, size=16)
    plt.ylabel('Meritvärde', size=16)
    plt.title("Korrelation av meritvärde vs %s" % query_var, size=18)
    plt.savefig(filename, format='svg')


# Make predictions for a new data point from the model trace
def query_model(trace, new_observation):
    # Print information about the new observation
    #print('New Observation')
    #print(new_observation)
    # Dictionary of all sampled values for each parameter
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Standard deviation
    sd_value = var_dict['sd'].mean()

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)

    # Align weights and new observation
    var_weights = var_weights[new_observation.index]

    # Means of variables
    var_means = var_weights.mean(axis=0)

    # Mean for observation
    mean_loc = np.dot(var_means, new_observation)

    # Distribution of estimates
    estimates = np.random.normal(loc=mean_loc, scale=sd_value, size=1000)

    """
    # Plot the estimate distribution
    #plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist=True, kde=True, bins=19,
                 hist_kws={'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws={'linewidth': 4},
                 label='Estimated Dist.')
    # Plot the mean estimate
    plt.vlines(x=mean_loc, ymin=0, ymax=5,
               linestyles='-', colors='orange', linewidth=2.5)
    plt.title('Density Plot for New Observation');
    plt.xlabel('Grade')
    plt.ylabel('Density')
    plt.show()

    # Estimate information
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                                             np.percentile(estimates, 95)))
    """

    return mean_loc


def db_connect(username, password, database, hostname="localhost"):
    try:
        db_connection = mariadb.connect(
            user=username,
            password=password,
            host=hostname,
            database=database
        )
    except mariadb.Error as e:
        print(f"Error: {e}")
        sys.exit(1)

    return db_connection


def get_dataframe(db_connection, db_query):
    db_cursor = db_connection.cursor()
    dataframe = pd.read_sql_query(db_query, db_connection)
    return dataframe


def clean_dataframe(dataframe):
    dataframe['sex_enum'] = np.where(dataframe['sex'] == "Kvinna", 0, 1)  # Kvinna = 0, Man = 1
    dataframe['language_enum'] = np.where(dataframe['mothertongue'] == "Svenska", 0, 1)  # Svenska = 0, annat språk = 1
    dataframe = dataframe.drop(columns=["sex", "mothertongue"])

    dataframe = dataframe[['ssn', 'sex_enum', 'language_enum',
                           'g6_sv', 'g6_sva', 'g6_en', 'g6_ma',
                           'g6_bi', 'g6_fy', 'g6_ke', 'g6_tk',
                           'g6_ge', 'g6_hi', 'g6_re', 'g6_sh',
                           'g6_bd', 'g6_hkk', 'g6_idh', 'g6_mu', 'g6_sl',
                           #'g6_mspr',
                           'g7_sv', 'g7_sva', 'g7_en', 'g7_ma',
                           'g7_bi', 'g7_fy', 'g7_ke', 'g7_tk',
                           'g7_ge', 'g7_hi', 'g7_re', 'g7_sh',
                           'g7_bd', 'g7_hkk', 'g7_idh', 'g7_mu', 'g7_sl',
                           #'g7_mspr',
                           'g8_sv', 'g8_sva', 'g8_en', 'g8_ma',
                           'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk',
                           'g8_ge', 'g8_hi', 'g8_re', 'g8_sh',
                           'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl',
                           #'g8_mspr',
                           'g9_sv', 'g9_sva', 'g9_en', 'g9_ma',
                           'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk',
                           'g9_ge', 'g9_hi', 'g9_re', 'g9_sh',
                           'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl',
                           #'g9_mspr'
                           ]]
    return dataframe


def main():
    # NumPy and Panda truncates arrays and dataframes by default
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Fetch a Panda dataframe with all rows and columns from the clean database table.
    conn = db_connect("fredrik", "fredrik", "fredrik", "localhost")
    df_all = get_dataframe(conn, "select * from clean")
    df_all = clean_dataframe(df_all)

    # Divide into dataframes suitable for regression. This makes code more readable later on.
    df_alumni = df_all[df_all["ssn"].str.match('^04.*|03.*|02.*|01.*|00.*|99.*|98.*|97.*|96.*') == True].copy()
    df_current = df_all[df_all["ssn"].str.match('^05.*') == True].copy()
    df_younger = df_all[df_all["ssn"].str.match('^08.*|07.*|^06.*') == True].copy()

    # Filter students who have somewhat complete grading, for ML purposes
    # Dropping alumni without grades in Ma and En results in a smaller dataset but enables
    # training the model with suitable student data instead of many null values
    g9_subjects_all = ['g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi',
                       'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl']
    df_younger.dropna(subset=['g6_ma', 'g6_en'], inplace=True)
    df_current.dropna(subset=['g8_ma', 'g8_en'], inplace=True)
    df_current.drop(columns=g9_subjects_all, inplace=True)
    df_alumni.dropna(subset=['g9_ma', 'g9_en', 'g8_ma', 'g8_en', 'g7_ma', 'g7_en', 'g6_ma', 'g6_en'], inplace=True)

    # Calculate merit value for all alumni based on the 16 subjects (17 including Sv/SvA)
    # We've excluded tertiary language because of too few samples from grade 6
    # 16 subjects (with grades) = 16 * 20 = 340 max
    df_alumni['meritvarde'] = df_alumni[g9_subjects_all].sum(axis=1)

    predict = 'meritvarde'

    # Order of labels for prediction:
    # g6_subj1 .. g6_subj17, g7_subj1 .. g7_subj17, g8_subj1 .. g8_subj17
    X = df_alumni.drop(columns=['ssn', 'sex_enum', 'language_enum', predict] + g9_subjects_all).fillna(0).copy()
    y = np.array(df_alumni[predict])

    # Evaluate all models by iterating and saving the best results based on mean average error to Pickle file
    X_train, X_test, y_train, y_test, model, mae = evaluate_models_iterate(X, y, repeat=2)

    # Load the best saved non-Bayesian model from the evaluation for predicting current student cohort
    saved_best_model = pickle.load(open(model_filename, 'rb'))
    saved_best_model.fit(X_train, y_train)
    model_predictions = saved_best_model.predict(df_current.drop(columns=['ssn', 'sex_enum', 'language_enum']).fillna(0))

    # Bayesian model
    # We use a custom formula since doing inference on all data points is slow (10 minutes). By using only grades
    # from year 8 we cut it down to ~50 seconds
    bayesian_formula = "meritvarde ~ g8_sv + g8_sva + g8_en + g8_ma + g8_bi + g8_fy + g8_ke + g8_tk + g8_ge + g8_hi + g8_re + g8_sh + g8_bd + g8_hkk + g8_idh + g8_mu + g8_sl"
    g8_subjects_all = ['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk',
                       'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl']

    normal_trace, pm = bayesian_model(bayesian_formula, df_alumni[['meritvarde'] + g8_subjects_all].fillna(0))
    df_current_bayesian = df_current[g8_subjects_all].fillna(0)

    df_current_bayesian.insert(0, 'Intercept', 1)
    df_current_bayesian['bayespredict'] = df_current_bayesian.apply(lambda row: query_model(normal_trace, row), axis=1)
    df_current_bayesian = df_current_bayesian[['bayespredict']]

    # Dataframe manipulation to get a CSV file with the last column as prediction
    df_current.reset_index(inplace=True, drop=True)
    df_current_bayesian.reset_index(inplace=True, drop=True)
    current_predictions = pd.DataFrame({'mv-guess-'+model: model_predictions})
    df_current = df_current.join(current_predictions)
    df_current = df_current.join(df_current_bayesian)
    df_current.to_csv('df_current_bestmodel.csv')


if __name__ == "__main__":
    main()

"""
# Keep in case we need it later
#df_full['total_points'] = df_full.drop(['ssn', 'sex', 'mothertongue', 'sex_enum', 'language_enum'], axis=1).sum(axis=1)

#cols = df_full.columns.tolist()
#cols.insert(1, cols.pop(cols.index('sex_enum')))
#cols.insert(2, cols.pop(cols.index('language_enum')))
#df_full = df_full.loc[:, cols]
#print(df_alumni.corr(method='pearson')['meritvarde'].sort_values())
#X = np.nan_to_num(np.array(df_alumni.drop(['ssn', 'sex_enum', 'language_enum', predict] + g9_subjects_all, 1)))
#bayesian_model(bayesian_formula, df_alumni.drop(columns=['ssn', 'sex_enum', 'language_enum'] + g9_subjects_all).fillna(0))
younger = df[df["ssn"].str.match('^06.*|^07.*') == True].copy()

eights = df[df["ssn"].str.match('^05.*') == True].copy()
eights.dropna(subset=['g8_ma'])

alumnis = df[df["ssn"].str.match('^04.*|03.*|02.*|01.*|00.*|99.*|98.*|97.*') == True].copy()
alumnis.dropna(subset=['g8_ma', 'g9_ma'], inplace=True)  # Only keep students who actually got a grade in year 8 and 9

g9_subjects_all = ['g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi', 'g9_re',
                   'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr']
g9_subjects_core = ['g9_sv', 'g9_sva', 'g9_en', 'g9_ma']

alumnis['meritvarde'] = alumnis[g9_subjects_all].sum(axis=1)  # 16 subjects + Mspr = 17 * 20 = 340 max
# plt.bar(alumnis['meritvarde'].value_counts().index, alumnis['meritvarde'].value_counts().values, fill='navy', edgecolor='k', width=1)
# plt.xlabel('Meritvärde')
# plt.ylabel('Antal elever')
# plt.yticks(np.arange(0, 21, 3))
# plt.title("Alumners distribution av meritvärde")
# plt.show()

# Grade distribution by address
# sns.kdeplot(alumnis.loc[df['language_enum'] == 0, 'meritvarde'], label = "Svenska", shade = True)
# sns.kdeplot(alumnis.loc[df['language_enum'] == 1, 'meritvarde'], label = "Annat", shade = True)
# plt.xlabel('Meritvärde'); plt.ylabel('Densitet'); plt.title('Densitetsplot av meritvärde och språk');
# plt.legend()
# plt.show()

# print(alumnis.corr()['meritvarde'].sort_values())
# print(alumnis[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl', 'g8_mspr', 'meritvarde']].corr()['meritvarde'].sort_values())

predict = "meritvarde"

# alumnis.drop(['ssn', 'sex', 'mothertongue', 'meritvarde', 'total_points', 'g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi', 'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr'], 1).info()

# X = np.nan_to_num(np.array(alumnis[['sex_enum', 'language_enum', 'g6_sv', 'g6_sva', 'g6_en', 'g6_ma', 'g7_sv', 'g7_sva', 'g7_en', 'g7_ma', 'g8_sv', 'g8_sva', 'g8_en', 'g8_ma']]))
# X = np.nan_to_num(np.array(alumnis.drop(['ssn', 'sex', 'mothertongue', 'meritvarde', 'g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi', 'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr'], 1)))
X = np.nan_to_num(np.array(alumnis.drop(
    ['ssn', 'sex', 'mothertongue', 'meritvarde', 'total_points', 'g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy',
     'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi', 'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr'],
    1)))
# X = np.nan_to_num(np.array(alumnis[['g6_sv', 'g6_sva', 'g6_en', 'g6_ma', 'g7_sv', 'g7_sva', 'g7_en', 'g7_ma', 'g8_sv', 'g8_sva', 'g8_en', 'g8_ma']]))
# X = np.nan_to_num(np.array(alumnis[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl', 'g8_mspr']]))
y = np.array(alumnis[predict])

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# results = evaluate_models(x_train, x_test, y_train, y_test)
# print(results)

linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)
best = 0

for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    results = evaluate_models(x_train, x_test, y_train, y_test)
    print(results)

    # Train the model and save it to Pickle file
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print(acc)

    if acc > best:
        best = acc

print("\nCoefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)
print("Best: ", best)

predictions = linear.predict(x_test)

# predictions_file = open("predictions2.txt", "w+")
for x in range(len(predictions)):
    # print(" Guess: {:.2f}".format(predictions[x]), " Real: ", y_test[x], "::", "Diff: {:.2f}".format(y_test[x]-predictions[x]))
    print(" Grades: ", format(x_test[x]), " Guess: {:.2f}".format(predictions[x]), " Real: ", y_test[x], "::",
          "Diff: {:.2f}".format(y_test[x] - predictions[x]))
#    print(",{:.2f}".format(predictions[x]), ",", y_test[x], ",{:.2f}".format(y_test[x] - predictions[x]), file=predictions_file)
# predictions_file.close()


eights.reset_index(inplace=True, drop=True)
# eights_predictions = linear.predict(eights[['g6_sv', 'g7_sv', 'g8_sv', 'g9_sv', 'g6_sva', 'g7_sva', 'g8_sva', 'g9_sva', 'g6_en', 'g7_en', 'g8_en', 'g9_en', 'g6_ma', 'g7_ma', 'g8_ma', 'g9_ma']].fillna(0))
# eights_predictions = linear.predict(eights[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl', 'g8_mspr']].fillna(0))
eights_predictions = linear.predict(eights.drop(
    ['ssn', 'sex', 'mothertongue', 'total_points', 'g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke',
     'g9_tk', 'g9_ge', 'g9_hi', 'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr'], 1).fillna(
    0))
eights_guess = pd.DataFrame({'g9_meritvarde_guess': eights_predictions})

eights_full = eights.join(eights_guess)
eights_full = eights_full[['ssn', 'g9_meritvarde_guess']]
eights_full.to_csv('eights_full.csv')

    # Code used before iterating on models and saving to Pickle file
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    results = evaluate_models(X_train, X_test, y_train, y_test)
    print(results.sort_values('mae'))
    
    model_linear = LinearRegression(); model_linear.fit(X_train, y_train)
    model_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train); model_elastic.fit(X_train, y_train)

    model_linear_predictions = model_linear.predict(df_current.drop(columns=['ssn', 'sex_enum', 'language_enum']).fillna(0))
    model_elastic_predictions = model_elastic.predict(df_current.drop(columns=['ssn', 'sex_enum', 'language_enum']).fillna(0))
    #current_predictions = pd.DataFrame({'mv-linear': model_linear_predictions, 'mv-elastic': model_elastic_predictions})
    
    #observation = pd.Series({'Intercept': 1, 'g8_sv': 10, 'g8_sva': 10, 'g8_en': 10, 'g8_ma': 10, 'g8_bi': 10, 'g8_fy': 10, 'g8_ke': 10, 'g8_tk': 10, 'g8_ge': 10, 'g8_hi': 10,
                       #'g8_re': 10, 'g8_sh': 10, 'g8_bd': 10, 'g8_hkk': 10, 'g8_idh': 10, 'g8_mu': 10, 'g8_sl': 10})
    #bayesian_prediction = query_model(normal_trace, observation)
"""

"""
normal_trace, pm = bayesian_model(bayesian_formula, df_alumni[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma',
                                                               'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi',
                                                               'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh',
                                                               'g8_mu', 'g8_sl', 'meritvarde']].fillna(0))


df_current_bayesian = df_current[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge',
                                  'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl']].fillna(0)
"""

"""
# Produce trace- and posterior plots
pm.traceplot(normal_trace)
fig = plt.gcf()
fig.savefig('traceplot.svg', format='svg')

pm.plot_posterior(normal_trace)
fig = plt.gcf()
fig.savefig('posteriorplot.svg', format='svg')
"""

"""
model_effect('g8_sv', normal_trace, df_alumni[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi',
                   'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl']].fillna(0))
"""