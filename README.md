# Hit Wave: A new approach to predicting Hit Songs!

## Code Overview

### Datasets
* **Datasets**: Initial Datasets to be merged
* **Preprocessed and Labeled Datasets**: Merged datasets after initial preprocessing and labeling
* **Preprocessed and Labeled Datasets with Spotify Features**: Merged datasets after initial preprocessing and labeling with Spotify features
* **Reordered Preprocessed and Labeled Datasets with Spotify Features**: Reordered merged datasets after initial preprocessing and labeling with Spotify features
* **Audio Features Datasets**: Datasets of features retrieved from the audio of the songs
* **Final Datasets without Filtering**: Final Datasets with Spotify and Audio Features
* **Final Datasets after Filtering**: Final Datasets with Spotify and Audio Features after initial filtering for removing duplicates
* **Final Datasets after Further Filtering**: Final Datasets with Spotify and Audio Features after initial filtering for removing duplicates and further filtering
* **EDA Datasets**: Exploratory Data Analysis Datasets containing statistics related to the final dataset utilized in the experiments: *Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_52896.csv*

### Initial Dataset 
* **Song_Dataset_Forming_and_Labelling.py**: Code for merging datasets in *Datasets* and labeling each song in the merged datasets as a hit. The merged and labeled datasets are saved into *Preprocessed and Labeled Datasets*.

### Spotify Information/Feature Extraction
* **Spotify_Track_ID_Extraction.py**: Code for extracting Spotify ID for each song in the datasets in *Preprocessed and Labeled Dataset*. 
* **Spotify_Audio_Feature_Extraction.py**: Code for extracting Spotify audio features by using Spotify ID for each song in the datasets in *Preprocessed and Labeled Dataset*. The datasets with the Spotify features are saved into *Preprocessed and Labeled Datasets with Spotify Features*.
* **Spotify_Track_Information_Extraction.py**: Code for extracting Spotify Information for each song in the datasets in *Preprocessed and Labeled Datasets with Spotify Features*. 
* **Reorder_Dataset.py**: Code for reordering dataset *Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv* resulting in the reordered dataset *Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv*.

### Audio Feature Extraction
* **Feature_Extraction.py**: Code for extracting several audio features from the audio of a song.
* **Audio_Retrieval_and_Feature_Extraction.py**: Code containing pipeline for retrieving the audio of each song in dataset *Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv* and saving the audio to *Retrieved_Audio* whilst retrieving features from it. The audio features are saved to *Audio Features Datasets\Audio_Features_Dataset.csv*.
* **Add_Audio_Features_to_Preprocessed_Dataset.py**: Code for merging the audio feature dataset, *Audio Features Datasets\Audio_Features_Dataset.csv*, with the reordered, pre-processed and labeled dataset with Spotify features *Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv* resulting in dataset *Final Datasets without Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_58425.csv*, dataset *Final Datasets after Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_56532.csv* after initial filtering, and dataset *Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_52896.csv* after further filtering.

### Models
* **Logistic_Regression.py**: Code for training LR Model
* **SVM_Linear_SVC.py**: Code for training SVM (Linear SVC) Model
* **Random_Forest.py**: Code for training RF Model
* **XGBoost.py**: Code for training XGBoost Model
* **DNN.py**: Code for training DNN Model

### Results
* **Regular Evaluation Results**: Datasets containing the results through regular evaluation of each model.
* **10-Fold Cross-Validation Results**: Datasets containing the results through 10-fold cross-validation evaluation of each model.
**Confusion Matrices**: Confusion matrices after performing 10-fold cross-validation evaluation of each model.

### Other
* **Exploratory_Data_Analysis.py**: Code for performing Exploratory Data Analysis on the final dataset used in the experiments *Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_52896.csv* resulting in dataset *EDA Datasets\Hits_n_No_Hits_per_Split.csv* containing the number of hits/non-hits per subset and bar plots saved to *Bar Plots*.
