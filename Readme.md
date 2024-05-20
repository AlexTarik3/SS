# PR(5) Multiclass classification

This project created in puprpose to clasifiy pets in pet shelter, based on their features characteristics. 
Main point of this project - create model which will help to predict how long animal will be in the shelter.


## Installation

To install the project, follow these steps:

1. Clone the repository: `git clone https://github.com/AlexTarik3/SS`
2. Navigate to the project file, in which you copied this repo
3. Check if you have Node.js and all libraries on your local device for start the application

## Usage

Data(train and new_input) should be placed in folder "data". They are creating there automatically after running file separate_data, but for this you need to change pathes all over each file in your drectory

In the pipeline folder you can find four files - train_model for training, test_model - for building predictions, additional one - preprocessing - for checking data before training and testing, and start. Start one is the one you need to run for training and testing models.
In the models you can see models - they`re created after training model and using it for testing and prediction.


## Models

In this project I used 4 models - 
- XGBoost;
- LightGBM;
- Random Forest;
- Extra Trees.

Most accurate for this type of classification was XGBoost, but others are totaly ok to use.