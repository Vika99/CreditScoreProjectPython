# CreditScoreProjectPython
The Project’s Goal: based on the arguments from the Dataset, determine ‘CreditScore’ value (Python)
Banks and such other financial lending institutions often need to look at a loan applicant’s credit history, economic status and such other factors to determine eligibility for loan, but the relationship between these factors is generally not well-defined and is most likely heuristic in nature. It is very often the case also that the company’s current standing, such as its rise or fall in the immediate past is taken into consideration to determine its financial stability. This might lead to erroneous judgement with regards to the company’s likelihood of defaulting on a loan. Using effective classification and time series analyses, we can generate a model that would not only be more precise but also cost-effective in solving this problem. 

It is worth noting that traditional credit scoring algorithms work linearly, analyzing historical data to provide estimates of future creditworthiness. In contrast, self-learning machine learning systems use historical and current data to improve their predictive capabilities. It is also possible to apply overfitting prevention and hyperparameter tuning techniques, which allows big data analysis. This will help establish relationships between disparate variables, giving a much deeper understanding of the borrower's profile. These features contribute to the steadily growing predictive potential of machine learning algorithms and better analysis of unstructured data. Machine learning algorithms are much more dynamic in self-updating, improving over time by discarding inefficient approaches and adding improvements without human intervention. This can be achieved by tracking down overfitting issues with cross-validation and then also using cross-validation for model selection.

Machine learning systems are quite cost-effective in the long run. Once these models are built, they can be used over and over again for all uses of credit. Most scorecard-based credit scoring solution providers charge users on a per-user basis. At the same time, machine learning models are a fully customizable and constantly learning system that can meet all your credit scoring and customer profiling needs. For example, ML ensemble models can provide flexible ML-based credit scoring systems that can provide accurate eligibility prediction and intelligent ranking of borrowers to minimize the number of potentially bad loans.

The available dataset was used from Kaggle (link: https://www.kaggle.com/datasets/parisrohan/credit-score-classification) that relates to a person’s credit-related information. The raw data includes information on 10,000 clients with 27 features. Below is the description of each column.

ID: Represents a unique identification of an entry;
Customer_ID: Represents a unique identification of a person;
Month: Represents the month of the year;
Name: Represents the name of a person
Age: Represents the age of the person;
SSN: Represents the social security number of a person;
Occupation: Represents the occupation of the person;
AnnualIncome: Represents the annual income of the person;
MonthlyInhandSalary: Represents the monthly base salary of a person;
NumBankAccounts: Represents the number of bank accounts a person holds;
NumCreditCard: Represents the number of other credit cards held by a person;
InterestRate: Represents the interest rate on a credit card;
NumofLoan: Represents the number of loans taken from the bank;
TypeofLoan: Represents the types of loan taken by a person;
Delayfromduedate: Represents the average number of days delayed from the payment date;
NumofDelayedPayment: Represents the average number of payments delayed by a person;
ChangedCreditLimit: Represents the percentage change in credit card limit;
NumCreditInquiries: Represents the number of credit card inquiries;
CreditMix: Represents the classification of the mix of credits;
OutstandingDebt: Represents the remaining debt to be paid (in USD);
CreditUtilizationRatio: Represents the utilization ratio of a credit card;
PaymentofMinAmount: Represents whether only the minimum amount was paid by the person;
TotalEMIpermonth: Represents the monthly EMI payments (in USD);
Amountinvestedmonthly: Represents the monthly amount invested by the customer (in USD);
PaymentBehaviour: Represents the payment behavior of the customer (in USD);
MonthlyBalance: Represents the monthly balance amount of the customer (in USD);
CreditHistoryYears: Represents the age of credit history of the person;

Identify Target Variable.
Based on the data exploration, our target variable appears to be CreditScore.

Data Cleaning is the most important part of any Machine Learning project. So, Initial data exploration reveals the following:

Many columns were with missing values. Given the high proportion of missing values, any technique to impute them will most likely result in inaccurate results.
It was also found that columns such as 'Age','NumofLoan','NumCreditInquiries','CreditHistoryYears' contained conflicting information that does not correspond to reality. And it was decided to replace these values with the average values for these columns. 
Certain static features not related to credit score, e.g., ID, Customer_ID , SSN, Name were dropped.
Statistical indicators, such as range of variation, median, and standard deviation, were considered to identify common patterns and trends to convert them into meaningful information.
The data was plotted on histograms to visually view the distribution of the data. The Pearson correlation was applied to consider the linear dependence of the variables.


The cleaned data has been saved to a new dataset 'NewDataFrame.csv'.

The transformer was used to clean up, shorten, and create functions. Functions have been created to convert categorical variables into dummy/indicator variables, using such tools as the Map function, get_dummies() function, and OneHotEncoder class.

- Map in Python is a function that works as an iterator to return a result after applying a function to every item of an iterable (tuple, lists, etc.).
-The get_dummies() function from the Pandas library can be used to convert a categorical variable into dummy/indicator variables.
-One-hot encoding is a process by which categorical data (such as nominal data) are converted into numerical features of a dataset.  The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features. The output will be a sparse matrix where each column corresponds to one possible value of one feature.

Based on domain knowledge, we will classify ‘CreditScore’ with the following ‘CreditScore’ values as 'Standard':0, 'Good':1,'Poor':2.

Next, there was implementation of division data into test and training, where necessary implementation of StandardScaler for Scaling and Standardization of values.

Execution of the following Algorithms were used in the project:
DecisionTreeClassifier 
RandomForest Classifier 
Adaptive boosting Classifier
GradientBoostingClassifier  
Voting Classifier
Support vector machines (SVM) 
FCNN(Fully Convolutional Neural Network).

In the following project were also used:

GridSearchCV. The technique for finding the optimal parameter values from a given set of parameters in a grid.

Feature selection. The classes in the sklearn.feature_selection module used for feature dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.

Automatic parameter searches, based on RandomizedSearchCV. randomly passes the set of hyperparameters and calculate the score and gives the best set of hyperparameters which gives the best score as an output, based on the estimator :RandomForestClassifier.


Based on Flask, a web application was created where a bank employee needs to substitute the client's parameters and, based on the model we trained and saved, we will see the result of the client's CreditScore: 'Standard':0, 'Good':1 or 'Poor':2.

