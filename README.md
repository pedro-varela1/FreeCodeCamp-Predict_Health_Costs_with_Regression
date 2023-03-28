# FreeCodeCamp-Predict Health Costs with Regression
My solution to FreeCodeCamp "Predict_Health_Costs_with_Regression" task: "In this challenge, you will predict healthcare costs using a regression algorithm.  You are given a dataset that contains information about different people including their healthcare costs. Use the data to predict healthcare costs based on new data."

## Dealing with data
After initializing the libraries and importing the data as a pandas dataframe, the first thing to do is analyze the data structure we are working with. So, to achieve this goal, we need to use methods like describe() and inspect the non-numeric features:

~~~python
print(dataset['sex'].value_counts(), '\n')
print(dataset['smoker'].value_counts(), '\n')
print(dataset['region'].value_counts())
~~~

Then, we convert the non-numerical data to numerical data:

~~~python
dataset_copy['sex'] = dataset['sex'].apply(lambda v: 1 if v=='male' else 0).astype(int)
dataset_copy['smoker'] = dataset['smoker'].apply(lambda v: 1 if v=='yes' else 0).astype(int)
dataset_copy['region'] = dataset['region'].apply(lambda v: 3 if v=='southeast' else 2 if v=='southwest' else 1 if v=='northwest' else 0).astype(int)
~~~

After further analysis, we normalize the features used in the analysis by applying:
~~~python
dataset_copy[v] = (dataset_copy[v] - min_value) / (max_value - min_value)
~~~

## The model
The most important aspect of creating a machine learning application is its modeling. For the analysis of health costs, two hidden layers with 64 neurons each were used, with the use of bias in one of the layers to accelerate the process, in addition to the RMSProp optimization function and the Mean Square loss function. The neural network is similar to a regression model, but much more accurate due to non-linearity.
The neural network was trained with 400 epochs.

~~~python
#Creation and compiling of the model
model = keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=[len(train_features.keys())]),
  layers.Dense(64, activation='relu', use_bias=True),
  layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.1)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])
              
#Training the model
history = model.fit(
  train_features, 
  train_labels,
  epochs=400, 
  validation_split = 0.2, 
  verbose=0)
~~~

## The result
The result reached the necessary parameters and a graph with the regression model (line) was plotted for comparison with the real points.
![baixados](https://user-images.githubusercontent.com/93870597/228094960-d54b5099-4de7-4d5a-b5e1-b6b7ed8077a5.png)


