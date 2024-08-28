"""
File: random_forest_model_wwom.py
Author: Dr. Andrew G. McLeish
Description: This script implements a Random Forest model to predict a label using operational taxonomic units (OTU) or environmental data.

For information on running the script see: README.md
"""
###########################################################################
# User input 
###########################################################################
# Filenames
env_data_filename = 'env_data_example.csv'
otu_table_filename = 'otu_table_example.csv'
label_data_filename = 'label_data_example.csv'

# Label name
label_name = 'label'

# Environmental variables
env_variable_list = [
    'env_var1',
    'env_var2', 
    'env_var3', 
    'env_var4', 
    'env_var5',
    'env_var6',
    'env_var7',
    'env_var8',
]

# OTU table
sample_ids = 'site_time'

# Taxa information columns
taxa_info_cols = 19

###########################################################################
# General imports
###########################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import joblib

###########################################################################
# Random forest code
###########################################################################
from sklearn.ensemble import RandomForestRegressor
# Model metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_absolute_error
# Modelling
from sklearn.model_selection import RandomizedSearchCV
# GroupKFold - Use this when you have data grouped into distinct sets, and you want to ensure that all samples from the same group are in the same fold.
from sklearn.model_selection import GroupKFold
np.random.seed(42)

###########################################################################
# Import Data
###########################################################################
# Read in csv files
env_data = pd.read_csv(os.path.join('.\input_files', env_data_filename))
otu_table = pd.read_csv(os.path.join('.\input_files', otu_table_filename))
label_data = pd.read_csv(os.path.join('.\input_files', label_data_filename))

###########################################################################
# Functions
###########################################################################
# Convert to proportional
def proportional(df):
    # Extract relevant columns for proportionalisation (exclude the first and last columns)
    cols_to_proportionalise = df.columns[1:-1]

    # Calculate the row sum for each row
    row_sums = df[cols_to_proportionalise].sum(axis=1)

    # Proportionalise by row
    df_proportionalised = df[cols_to_proportionalise].div(row_sums, axis=0)

    # Combine the ID column with the proportionalised data
    df_result = pd.concat([df[df.columns[0]], df_proportionalised, df[df.columns[-1]]], axis=1)

    return df_result

# Grouping the data
def grouping(df):
    df_grouped = df.copy()
    df_grouped[sample_ids] = df_grouped[sample_ids].apply(lambda x: x.split('_')[0])
    return df_grouped

# Phylogenetic resolution function
def phylogenetic_resolution(classification, df):
    df_classification = df.iloc[:,taxa_info_cols:]
    df_classification[classification] = df[classification]

    # Step 1: Create an empty dictionary
    classification_dict = {}

    # Step 2: Loop through each row of the dataframe
    for i, row in df_classification.iterrows():
        # Step 3: Check if the value in the last column of the current row already exists in the dictionary
        if row.iloc[-1] in classification_dict:
            # Step 4: If it does, add the values in the other columns to the existing values in the dictionary
            classification_dict[row.iloc[-1]] = [classification_dict[row.iloc[-1]][j] + row.iloc[j] for j in range(len(row)-1)]
        else:
            # Step 5: If it doesn't, add a new key-value pair to the dictionary
            classification_dict[row.iloc[-1]] = row.iloc[:-1].tolist()

    # Step 6: Create a new dataframe from the dictionary
    df_classification_output = pd.DataFrame.from_dict(classification_dict, orient='index', columns=df_classification.columns[:-1])
    
    # Step 7: Rename the index column
    df_classification_output = df_classification_output.rename_axis(f'{classification}')

    return df_classification_output

# Running the random forest model
def random_forest_model(model_name, data_type, phylo_select, group, n_trees):
    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    ############
    # Select the dataframe type, biotic 'raw' reads/'pro'portional or 'env'ironmental
        # raw = biotic data raw reads
        # pro = biotic data proportional
        # env = environmental data
    if data_type == 'raw' or data_type == 'pro':
        df = otu_table
    elif data_type == 'env':
        df = env_data
        variable_list = env_variable_list
    else:
        print('Select data_type as \'raw\', \'pro\' or \'env\'.')
    
    if phylo_select == 'species' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    elif phylo_select == 'genus' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    elif phylo_select == 'family' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    elif phylo_select == 'order' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    elif phylo_select == 'class' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    elif phylo_select == 'phylum' and data_type != 'env':
        df = phylogenetic_resolution(phylo_select, df)
        df = df.T
        df.reset_index(inplace=True)
        df.rename(columns={'index':sample_ids}, inplace=True)
        if data_type == 'pro':
            df = proportional(df)
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        else:
            df =  pd.merge(df, label_data, on=sample_ids, how='left')
        variable_list = df.columns[1:-1]

    if data_type == 'pro':
        df = proportional(df)

    if group == 'yes':
        df = grouping(df)
    elif group == 'no':
        df = df
    else:
        print('Select group as \'no\' or \'yes\'.')
    ##############
    print(f'Model name: {model_name}')
    print(f'Data type: {data_type}')
    print(f'Grouping: {group}')
    print(f'Number of trees: {n_trees}')
    ##############
    # Target and features
    target = df[label_name] 
    features = df[variable_list]

    # Create NumPy arrays for features and target
    features = np.array(features)
    target = np.array(target)

    # Keep features as a dataframe
    features_df = df[variable_list]  # Select the relevant columns
    # Create a mapping from numeric indices to feature names
    index_to_feature_name = dict(enumerate(features_df.columns))

    # Split data by site - GroupKFold object
    group_kfold = GroupKFold(n_splits=5) # n_splits indicates the number of folds to split the data, cross-validation (commonly 5 or 10)
    # Define group variable for split
    split_group = df[sample_ids]

    # Assuming 'features', 'target', and 'groups' are your data
    for train_index, test_index in group_kfold.split(features, target, split_group):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

    # Split ratio
    split_ratio = len(X_test) / len(features)
    print('Split ratio: {:.2f}'.format(split_ratio))

    ###########################################################################
    # Random Forest parameters
    ###########################################################################
    # Define the hyperparameter grid for Random Forest
    param_dist = {
        'n_estimators': [n_trees],  # Number of trees
        'max_features': [1.0, 'sqrt', 'log2'],  # Number of features to consider at each split
        'min_samples_split': np.arange(2, 11),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': np.arange(1, 11)  # Minimum number of samples required to be a leaf node
    }

    # Create a Random Forest regressor
    rf_regressor = RandomForestRegressor(random_state=42)

    # Create a RandomizedSearchCV object for hyperparameter tuning
    tune_result = RandomizedSearchCV(
        rf_regressor, 
        param_distributions=param_dist, 
        n_iter=200, 
        cv=5, 
        n_jobs=-1, 
        random_state=42,
        verbose=2)

    # Progress bar output
    print('Starting random forest training...')
    
    # Fit the RandomizedSearchCV on your data
    tune_result.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = tune_result.best_params_
    print(f'Best parameters: {best_params}')

    # Get the best Random Forest model with the best hyperparameters
    best_rf_model = tune_result.best_estimator_

    # Make the prediction
    y_pred = best_rf_model.predict(X_test)

    ###############
    # Score Metrics
    #==============
    # # Accuracy calculation - not compatible with regression continuous value
    # accuracy_class = accuracy_score(Y_test_flt, y_pred_flt)
    # print("Accuracy:", f'{accuracy_class:.4f}')

    # Mean squared error calculation
    mse_regress = mean_squared_error(y_test, y_pred)
    print('Mean squared error:', f'{mse_regress:.4f}')

    # Root mean squared error - sqrt of MSE, provides a measure of the spread of errors
    rmse_regress = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Root mean squared error:', f'{rmse_regress:.4f}')

    # R^2 calculation
    r2_regress = r2_score(y_test, y_pred)
    print('R squared:', f'{r2_regress:.4f}')

    # Mean absolute error
    mae_regress = mean_absolute_error(y_test, y_pred)
    print('Mean absolute error:', f'{mae_regress:.4f}')

    # Explained variance score - quantifies the proportion of the variance in the dependent variable that is predictable from the independent variables
    ex_var_regress = explained_variance_score(y_test, y_pred)
    print('Explained variance score:', f'{ex_var_regress:.4f}')

    # Score metric dataframe
    score_metrics_df = pd.DataFrame(columns = ['Metric', 'Score'])
    score_metrics_df.loc[0] = ['Mean squared error', f'{mse_regress:.4f}']
    score_metrics_df.loc[1] = ['Root mean squared error', f'{rmse_regress:.4f}']
    score_metrics_df.loc[2] = ['R squared', f'{r2_regress:.4f}']
    score_metrics_df.loc[3] = ['Mean absolute error', f'{mae_regress:.4f}']
    score_metrics_df.loc[4] = ['Explained variance score', f'{ex_var_regress:.4f}']

    # Feature Importances
    #====================
    # Get the feature importances
    feature_importances = best_rf_model.feature_importances_

    # Create a DataFrame to associate feature importances with feature names
    feature_importance_df = pd.DataFrame({'Feature': range(X_test.shape[1]), 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Map the numeric indices to feature names in feature_importance_df
    feature_importance_df['Feature'] = feature_importance_df['Feature'].map(index_to_feature_name)
    # Print the feature importances
    print(feature_importance_df)

    # Copies dataframe to clipboard
    feature_importance_df.to_clipboard(index=False, sep='\t')

    #######################
    # Log output
    #######################
    # Output score metrics and feature importance information and print process
    score_metrics_df.to_csv(f'log_files\\{model_name}_score_metrics.csv', index = False)
    feature_importance_df.to_csv(f'log_files\\{model_name}_feature_importance.csv', index = False)

    print(f'Output score metrics file: log_files\\{model_name}_score_metrics.csv')
    print(f'Output feature importance file: log_files\\{model_name}_feature_importance.csv')

    #######################
    # Plots
    #######################
    # Prediction plot
    #======================
    # Assuming you have Y_test_flt and y_pred_flt
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel(f'True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction Plot for {model_name}')

    # Calculate the line of best fit (regression line)
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='red')

    # Set the same limits for both axes
    min_value = min(np.min(y_test), np.min(y_pred))
    max_value = max(np.max(y_test), np.max(y_pred))
    plt.xlim(min_value-0.1, max_value+0.1)
    plt.ylim(min_value-0.1, max_value+0.1)
    plt.savefig(f'plots\\{model_name}_prediction_plot.png', dpi=400)

    # Bar graph of importances
    #=======================
    # Plot the bar graph with the updated feature names
    if len(feature_importance_df) <= 30:
        feature_importance_df.sort_values(by='Importance', ascending=True).plot(x='Feature', y='Importance', kind='barh')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature importances for {model_name}')
    else:
        feature_importance_df.iloc[:30].sort_values(by='Importance', ascending=True).plot(x='Feature', y='Importance', kind='barh')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature importances for {model_name}')
    plt.tight_layout()
    plt.savefig(f'plots\\{model_name}_feature_importances.png', dpi=400)

    end_time = datetime.now()
    print('End time:', end_time.strftime("%Y-%m-%d %H:%M:%S"))
    # Get the total number of seconds in the timedelta object
    total_seconds = (end_time - start_time).total_seconds()

    # Calculate the total number of minutes and seconds
    total_minutes = int(total_seconds // 60)
    remaining_seconds = int(total_seconds % 60)

    # Output log file
    with open(f'log_files\\{model_name}_best_params_df.txt', 'w') as file:
        file.write(f'Log file for random_forest_model_wwom.py script\n')
        file.write(f'Start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")} \n')
        file.write(f'\n')
        file.write(f'\n')
        file.write(f'Random forest model settings\n')
        file.write(f'Model name: {model_name}\n')
        file.write(f'Data type: {data_type}\n')
        file.write(f'Phylogenetic selection: {phylo_select}\n')
        file.write(f'Grouping: {group}\n')
        file.write(f'Number of trees: {n_trees}\n')
        file.write(f'\n')
        file.write('Split ratio: {:.2f}'.format(split_ratio))
        file.write(f'\n')
        file.write(f'\n')
        file.write(f'Best parameters\n')
        for key, value in best_params.items():
            file.write(f'{key}: {value}\n')
        file.write(f'\n')
        file.write(f'Metric scores:\n')
        file.write('Mean squared error: ' + f'{mse_regress:.4f}')
        file.write(f'\n')
        file.write('Root mean squared error: ' + f'{rmse_regress:.4f}')
        file.write(f'\n')
        file.write('Mean absolute error: ' + f'{mae_regress:.4f}')
        file.write(f'\n')
        file.write('Explained variance score: ' + f'{ex_var_regress:.4f}')
        file.write(f'\n')
        file.write(f'\n')
        file.write(f'End time: {end_time}\n')
        file.write(f'Total run time: {total_minutes} mins and {remaining_seconds} seconds.\n')
        print(f'Random forest model completed.\n Total run time: {total_minutes} mins and {remaining_seconds} seconds.')
    trained_features = features_df.columns.tolist()
    return joblib.dump(best_rf_model, f'model_settings\\{model_name}_model_parameters.pkl'), joblib.dump(trained_features, f'model_settings\\{model_name}_model_features.pkl')

###########################################################################
# Command line arguments
###########################################################################
def main(args):
    random_forest_model(
        args.model_name,
        args.data_type,
        args.phylo_select,
        args.group,
        args.n_trees
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest model for biotic or environmental data.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model.")
    parser.add_argument('--data_type', type=str, required=True, choices=['raw', 'pro', 'env'], help="Type of data (e.g., 'raw', 'pro', 'env').")
    parser.add_argument('--phylo_select', type=str, choices=['genus', 'family', 'order', 'class', 'none'], help="Phylogenetic selection for biotic data: 'genus', 'family', 'order', 'class'. If using environmental data: 'none'.")
    parser.add_argument('--group', type=str, required=True, choices=['yes', 'no'], help="Grouping is applied to the 0th index string after splitting the sample ID by '_'. E.g., if sample ID is AVOND_cp10, then all AVOND are grouped together. Select 'no' for no grouping.")
    parser.add_argument('--n_trees', type=int, required=True, help="Number of trees.")

    args = parser.parse_args()

    # Automatically set phylo_select to 'none' if data_type is 'env'
    if args.data_type == 'env':
        args.phylo_select = 'none'

    main(args)