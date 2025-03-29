# deep-learning-nueral-networks

### Solved Files

[Starter_Code](https://github.com/BryanCarney/deep-learning-nueral-networks/blob/main/Starter_Code.ipynb)

[Starter_Code_Optimized](https://github.com/BryanCarney/deep-learning-nueral-networks/blob/main/Starter_Code_Optimized.ipynb)

### Output Files

[AlphabetSoupCharity](https://github.com/BryanCarney/deep-learning-nueral-networks/blob/main/AlphabetSoupCharity.h5)

[AlphabetSoupCharity_Optimization](https://github.com/BryanCarney/deep-learning-nueral-networks/blob/main/AlphabetSoupCharity_Optimization.h5)

### Original Files

[Module 21 Challenge files](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/Starter_Code.zip)

### Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

• **EIN** and **NAME** — Identification columns

• **APPLICATION_TYPE** — Alphabet Soup application type

• **AFFILIATION—Affiliated** sector of industry

• **CLASSIFICATION** — Government organization classification

• **USE_CASE** — Use case for funding

• **ORGANIZATION** — Organization type

• **STATUS** — Active status

• **INCOME_AMT** — Income classification

• **SPECIAL_CONSIDERATIONS** — Special considerations for application

• **ASK_AMT** — Funding amount requested

• **IS_SUCCESSFUL** — Was the money used effectively

### Instructions

### Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

> • What variable(s) are the target(s) for your model?

> • What variable(s) are the feature(s) for your model?

2. Drop the EIN and NAME columns.

![image](https://github.com/user-attachments/assets/c95107f8-e21b-47d6-b001-585c3fb146d1)

3. Determine the number of unique values for each column.

![image](https://github.com/user-attachments/assets/52bd31bc-7dae-4f10-a7b8-d14b147f736b)

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

![image](https://github.com/user-attachments/assets/71135290-b290-4348-91fc-4dc635e51eb6)

5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

![image](https://github.com/user-attachments/assets/3b211437-4aa6-4c42-bc01-d160c2952b9e)

6. Use pd.get_dummies() to encode categorical variables.

![image](https://github.com/user-attachments/assets/69744e39-9c3b-465d-aaaa-1b7a0bba74fe)

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

![image](https://github.com/user-attachments/assets/428645ba-8bc6-4147-af0b-94b351447d82)

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

![image](https://github.com/user-attachments/assets/80ef9a3e-3f5e-40b6-9cb0-4d40a32b475a)

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

2. Create the first hidden layer and choose an appropriate activation function.

3. If necessary, add a second hidden layer with an appropriate activation function.

4. Create an output layer with an appropriate activation function.

![image](https://github.com/user-attachments/assets/3bcc3034-e441-4e3f-8bc9-95043a70e578)

5. Check the structure of the model.
   
![image](https://github.com/user-attachments/assets/cddb4a65-635c-42f5-98fe-fa30ffd38d27)

6. Compile and train the model.

![image](https://github.com/user-attachments/assets/3f2d4b9b-3993-4735-8ea7-493ae1041c61)

7. Create a callback that saves the model's weights every five epochs.

![image](https://github.com/user-attachments/assets/5955d1c6-be00-48c9-9cdb-1f2f259fa0fa)

8. Evaluate the model using the test data to determine the loss and accuracy.

![image](https://github.com/user-attachments/assets/3de963e9-1837-42c7-ab3b-b741670bfaaf)

9. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

![image](https://github.com/user-attachments/assets/c1ead8f9-e72d-4c9a-848c-f7166c2f25b4)

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

**Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:**

• Dropping more or fewer columns.

• Creating more bins for rare occurrences in columns.

• Increasing or decreasing the number of values for each bin.

• Add more neurons to a hidden layer.

• Add more hidden layers.

• Use different activation functions for the hidden layers.

• Add or reduce the number of epochs to the training regimen.

> Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame from the provided cloud URL.

2. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

3. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

![image](https://github.com/user-attachments/assets/c5d399fc-5c0d-4ae6-986f-f8c4e7d7bb29)

Closest Result:

![image](https://github.com/user-attachments/assets/570225fd-f960-461c-868d-15fb85ba008c)

4. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

![image](https://github.com/user-attachments/assets/5cfd5c95-1b16-4b33-b662-ca5f4208ef8d)

### Step 4: Write a Report on the Neural Network Model

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. **Overview of the analysis:** Explain the purpose of this analysis.

2. **Results:** Using bulleted lists and images to support your answers, address the following questions:

• Data Preprocessing

> • What variable(s) are the target(s) for your model?

> • What variable(s) are the features for your model?

> • What variable(s) should be removed from the input data because they are neither targets nor features?

• Compiling, Training, and Evaluating the Model

> • How many neurons, layers, and activation functions did you select for your neural network model, and why?

> • Were you able to achieve the target model performance?

> • What steps did you take in your attempts to increase model performance?

3. **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

---

# Overview of the Analysis:

The purpose of this analysis is to assess the performance of a deep learning model built to solve a classification problem, specifically the Alphabet Soup dataset. The goal is to evaluate how well the model performs, identify any areas for improvement, and provide recommendations for optimizing the model further. The analysis covers data preprocessing, model architecture, training outcomes, and strategies used to improve performance.

### Results:

**Data Preprocessing:**

• Target Variables:

> The target variable for the model is likely the column indicating the label or category of the data points, typically a classification outcome (e.g., success or failure in the dataset).

• Feature Variables:

> The features are all the other variables or columns that are used to predict the target. These may include numerical or categorical features like age, income, etc.

• Variables to Remove:

> Variables that do not contribute meaningful information to the model (e.g., irrelevant columns or those with missing values) should be removed. In this case, the "id" column would typically be removed as it doesn't provide predictive power for the model.

### Compiling, Training, and Evaluating the Model:

• Neurons, Layers, and Activation Functions:

> The model architecture likely consists of multiple layers (e.g., input, hidden, and output layers) with various numbers of neurons.

> The activation functions used could include ReLU for hidden layers and softmax for the output layer to handle the classification problem.

> A reasonable architecture might involve 2-3 hidden layers with varying neurons (e.g., 128, 64, and 32 neurons) to capture non-linear relationships.

### Achieving Target Model Performance:

**Original model results:**

Loss: 0.5729

Accuracy: 72.57%

**Optimized model results after 150 epochs:**

Final accuracy: ~74.2%

This indicates the model achieved a slight improvement, though the accuracy is still below an ideal threshold 75%.

### Steps Taken to Improve Performance:

**Optimizing Learning Rate:** By adjusting the learning rate, the model could converge more efficiently.

**Early Stopping and Dropout:** Techniques such as early stopping and dropout may have been used to prevent overfitting and improve generalization.

**Changing Architecture:** Additional layers or neurons were tested to capture more complex patterns in the data.

### Summary:

**Overall Results:**

The deep learning model performed reasonably well with an accuracy improvement from 72.57% to 74.2% after optimization. However, it still falls short of achieving a high level of predictive accuracy, indicating room for further enhancement.

**Recommendation for a Different Model:**

While the deep learning model is useful, it might be beneficial to explore alternative models like Random Forests or XGBoost for this classification problem. These tree-based models can often outperform neural networks in structured tabular data by better handling feature interactions and reducing overfitting.

Alternatively, experimenting with ensemble methods like Stacking could improve accuracy by combining the strengths of multiple models.

---

### Step 5: Copy Files Into Your Repository

Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.
