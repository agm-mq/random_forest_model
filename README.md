# Random Forest Model for Wet Weather Overflow Monitoring
## Description
File: `random_forest_model_wwom.py`

Author: Dr. Andrew G. McLeish

Description: This script implements a Random Forest model to predict a label using operational taxonomic units (OTU) or environmental data. 

## Cross-validation and hyperparameter tuning
`GroupKFold` is used with 5 splits to handle cross-validation.
 
Hyperparameter tuning is performed to optimise the Random Forest model; the following parameters are tuned:

Number of Trees (`n_trees`): Define the number of trees in the forest.

Max Features (`max_features`): Number of features to consider at each split. Options include `1.0`, `sqrt`, and `log2`.

Minimum Samples Split (`min_samples_split`): Minimum number of samples required to split an internal node. Values range from 2 to 10.

Minimum Samples Leaf (`min_samples_leaf`): Minimum number of samples required to be a leaf node. Values range from 1 to 10.

## Output 
The script will output files in the log_files, plot_files, and model_settings folders.
### \log_files
- Score metrics (e.g., R<sup>2</sup> score)
- Feature importances
- Best hyperparameters
### \plot_files
- Plots of the predictions vs. true values
- Plots of feature importances
### \model_settings
- Model features
- Model parameters

## Setting up the script
Install modules if required using the 'pip install' command, such as:

    pip install scikit-learn
 
Import the environmental, OTU table, and label data files into the 'input_files' folder and check that the format is the same as the example files.

Check the `env_data_filename`, `otu_table_filename`, and `label_data_filename` variables in the script to match the filenames of the files that were imported.

Check the `label_name` is correct.

Check the environmental variables in `env_variable_list`.

Check the `sample_ids` so it matches with the sample IDs in all three files.

Check the number of `taxa_info_cols` so it matches the number of information columns in the OTU table file.

These variables are located in lines 16-44 as shown below:
```
###########################################################################
# User input 
###########################################################################
# Filenames
env_data_filename = 'env_data_example.csv'
otu_table_filename = 'otu_table_example.csv'
label_data_filename = 'label_data_example.csv'

# Label name
label_name = 'label'

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

```
## Usage:
### Open Command Prompt
- Press 'Win + R', type `cmd`, and press enter
### Navigate to the directory
- Use the `cd` command to change the directory to where the Python script is located. For example:

      cd Documents\GitHub\random_forest_model
### Run the Python script
- Type `python` followed by the script name and options:

      python random_forest_model_wwom.py --model_name --data_type --phylo_select --group --n_trees

## Options

### `--model_name`

- `model_name1`: This will be used as part of the filenames in the output files.

### `--data_type`

- `raw`: Uses the raw read counts in the OTU table.
- `pro`: Proportionalises the count data in the OTU table.
- `env`: Uses the environmental data.

### `--phylo_select` 

- `genus`: Selects genus-level data.
- `family`: Selects family-level data.
- `order`: Selects order-level data.
- `class`: Selects class-level data.
- `none`:  For environmental data.

### `--group`

- `yes`: Grouping is applied to the 0th index string after splitting the sample ID by '_'. E.g., if sample ID is site1_cp1, then all site1 are grouped together.
- `no`: No grouping.

### `--n_trees`

- `1000`: The number of trees in the random forest.
## Example runs
Example 1: 

`python random_forest_model_wwom.py --model_name model_run1 --data_type raw --phylo_select genus --group yes --n_trees 10`

Example 2: 

`python random_forest_model_wwom.py --model_name model_run2 --data_type env --phylo_select none --group yes --n_trees 10`