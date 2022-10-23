# Generate datasets

**Note** : Run all the scripts from the root directory adn_circulant

First check all the data (metadata data and raw data) are present in the directory data/raw

Then run the following command to format data and merged raw data with metadata; it will create 3 files:
 - merged/diag_data.csv
 - merged/chir_data.csv
 - merged/end_data.csv
```
python data_processing/load.py
```

Then you need to generate features and labels using the script compute_features.py.

This script takes 3 arguments (they all have default values), and you can find information about these parameters in the code or by executing the command :
```
python data_processing/compute_features.py --help
```