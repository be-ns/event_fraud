## FRAUD DETECTION FOR EVENTS
### Predicting the probability of fraud for Event Data.
#### [Benjamin Siverly](github.com/be-ns), [Parker Stevens](github.com/pstevens33), [Lindsey Eggleston](github.com/lindseyeggleston), [Tim Kahn](github.com/timkahn)
-------
#### __Basic Overview__  
Done as a case study in part with the [Galvanize Data Science Immersive](https://www.galvanize.com/denver-platte/data-science).
Data was gathered from confidential source and is not available for public viewing.  

Our [Web App](http://ec2-52-70-42-234.compute-1.amazonaws.com:8150/) allows users to interact with out Fraud detection model but inputting the costs / benefits associated with a Fraudulent act, as well as a threshold for action. After doing this, a new data point is pulled from Heroku, passed through our [model](github.com/be-ns/event_fraud/gradient_boosting.py) to receive a probability of the event being fraudulent or not. The data and the prediction are stored in a MongoDB, and the output is shown with a expected profit based on the threshold combined with the costs for action given by the user. 

Initial data was grouped into four subcategories which were then munged and transformed by each party using EDA. After building a simple Machine Learning model, the top 3 features from each model were passed along to the main [script](github.com/be-ns/event_fraud/collect_data.py), which compiles all top features into a single DF.

Initial modeling was done with Gradient Boosting and Logistic Regression. Gradient Boosting for model accuracy and Logistic Regression for beta analysis. 

Our web app allows a user to input the cost/benefit for fraud (benefit to catching fraudulent events, cost of investigating an event that is not fraudulent) and outputs a profit curve with an ideal threshold for risk that maximized the profit.
The results were found using our [Gradient Boosted model](github.com/be-ns/gradient_boosting.py). [F1 score](https://chrisalbon.com/machine-learning/precision_recall_and_F1_scores.html) and Accuracy are given for the model for predicted outputs, not for the user-inputted parameters. 

#### __Method__
We Utilized the [CRISP-DM](https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining) method for Data Mining and time organization.  

_Data Understanding_:
Part of the difficulty of this project was the lack of clarity on the specific features we were using. There were categorical variables (such as user-type), numeric variables (like tickets sold), Booleans, HTML, and JSON strings. To build a working model quickly we ignored the HTML and JSON features and focused on cleaning and munging the numerics and categoricals, which were then used in flexible models like Random Forests and Logistic Regression. 
We defined `Fraud` in this sense as any categorical that listed the words fraud in the titl itself. This excluded titles like `spam`, `spammer`, and `tos_warn`. Without proper education on the types of fraud we were unable to decide objectively if a `spam` event was in the same category as a `fraud` event. This definition could easily be changed if further information yields new insight. 

_Data Preparation_:  
Initial EDA and scripting was done in Pandas. Care was taking to ensure balanced results by (`stratifying`)[http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html] results in our train/test/split, and weighting the classes in our model.  
We engineered the following features:
* `Tickets Left` - Took the number of available tickets and subtracted the number of tickets sold.
* `Types Tickets` - How many different ticket types were advertised. 1 indicates a 'cover charge', while 7 or 8 indicates tiered quality for event types
* `User Class` - Leave-One-Out dummy-ized variables for 4 of the 5 user types.
* `Created to Start` - Utilized timestamp data to measure the elapsed time between when the event was created and when it started.

Special care was taken to ensure all variables were in the correct format (no `Boolean`s were left as `int64`'s). All data was cleaned using Pandas & Python 3.

_Modeling / Evaluation_:  
We used F1-score and three fold cross validation to measure the accuracy of the model, since we had unbalanced classes. Normal accuracy meausures would be pointless, since we had less than 10% of data being categorized as Fraudulent. Class balancing was the most vital aspect for our model to be accurate. 

Two models were built: Logistic Regression and a Gradient Boosted Tree.  
Logistic Regression is a nice baseline, and gives a clean, interpretable Beta for each feature to interpret the model.  
Gradient Boosting provides a highly accurate and swift way of splitting the data on certain metrics (for instance, a `Boolean` would be split on 0 or 1) then returning to weight certain features using [Gradient Descent](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwi5y9WlqJvUAhWmslQKHTIUDMwQFggwMAI&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGradient_descent&usg=AFQjCNEB7szBsRwTf-gol1jdLEMPb9r-UA&sig2=jCVOR1slNfzk6rMSgOqYaQ).


Our largest Betas from Logistic Regression:
1. `Tickets Left` : `6.3` - (using scaled data) - Implies that the more tickets left at the time of the event means the event is more likely to be fraudulent. 
2. `Previous Payouts` : `-8.3` - (using scaled data) - Implies that the previous payouts (number of events the host has held) were negatively correlated (decreased payouts means increased probability of fraud)

Our most important features for Gradient Boosting Classifier:
1. `Tickets Left` : The number of unsold tickets (`available tickets minus tickets sold`) was the strongest fraud predictor.
2. `Body Length` : The length that of the advertisement text for the event was our second-best predictor.

_Business Understanding_:
Ultimately the potentialfor fraud was not simply a yes or no. We decided to use a threshold for business action. Since False Negatives (missed fraud) was more risky than False Positives (overly cautious) we wanted to set our initial threshold for action low. We used a Profit Curve to estimate the optimal threshold to increase benefits.  

For our web app, a user can input different cost/benefit breakdowns and see the resulting profit curve, then set a recommended threshhold for action. From there, we would run a new data point through our model and output the probability of it being fraud or not. This act would yield an estimated cost for the data point.  
This cost was calculated in the following way:
Expected Profit / Loss = `(P(fraud) * (benefit of catching fraud - cost of investigating fraud)) - ((1-P(fraud)) * (cost of investigating fraud))`


#### __Technology Stack__
1. [Python](https://www.python.org/)
2. [Pandas](http://pandas.pydata.org/pandas-docs/stable/)
3. [SKLearn](http://scikit-learn.org/stable/)
4. [Numpy](https://docs.scipy.org/doc/numpy/reference/)
5. [Flask](http://flask.pocoo.org/docs/0.12/)
6. [AWS (EC2 and S3)](https://aws.amazon.com/)
7. [Seaborn](https://seaborn.pydata.org)
8. [MatPlotLib](matplotlib.org)
9. [JQuery](https://jquery.com/) * 

#### [__Flask App__](github.com/be-ns/event_fraud/gradient_boosting.py)


#### __Profit Curves__
At each possible threshold for flagging possibly-fraudulent events, profit per event was calculated as:
`(cost of investigating - cost of missing fraud)x(True Positive Rate) + (cost of investigating)x(False Positive Rate)`
We dynamically calculated the profit curve by iterating over all thresholds, given the cost of missing fraud and cost of investigating fraud from a user.  Final results were displayed as profit per 10,000 events.






* All JQuery done by [Parker Stevens](github.com/pstevens33)
