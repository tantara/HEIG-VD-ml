import csv
import random
from sklearn import naive_bayes
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.linear_model.ridge import Ridge
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

#Set the seed
random.seed(42)

'''
df=pd.read_csv('./data/ech_apprentissage.csv',";")
df.head()

df["prime_tot_ttc"].describe()
df_groupby=df[["var14","prime_tot_ttc"]].groupby("var14").mean()
ax=df_groupby.plot(kind="bar")
fig = matplotlib.pyplot.gcf()
ax.set_title('       Title \
                     \n Sub Title',fontsize=16)
ax.set_ylabel('Y label - %')
fig.set_size_inches(15.0, 10.0)

df["prime_tot_ttc"].describe()
df_groupby=df[["anc_veh","prime_tot_ttc"]].groupby("anc_veh").mean()
ax=df_groupby.plot(kind="bar")
fig = matplotlib.pyplot.gcf()
ax.set_title('       Title \
                     \n Sub Title',fontsize=16)
ax.set_ylabel('Y label - %')
fig.set_size_inches(15.0, 10.0)

#######################################################################################################

df["prime_tot_ttc"].describe()
df_groupby=df[["energie_veh","prime_tot_ttc"]].groupby("energie_veh").mean()
ax=df_groupby.plot(kind="bar")
fig = matplotlib.pyplot.gcf()
ax.set_title('       Title \
                     \n Sub Title',fontsize=16)
ax.set_ylabel('Y label - %')
fig.set_size_inches(15.0, 10.0)
###
'''

# Open the file
filepath_train="./data/ech_apprentissage.csv"
file_open=open(filepath_train,"r")
# Open the csv reader over the file
csv_reader=csv.reader(file_open,delimiter=";")
# Read the first line which is the header
header=csv_reader.next()
# Load the dataset contained in the file

marque = []
job = []

dataset=[]
for row in csv_reader:
    try:
        mi = marque.index(row[3])
    except:
        marque.append(row[3])
        mi = marque.index(row[3])
    row[3] = mi

    try:
        mi = job.index(row[10])
    except:
        job.append(row[10])
        mi = job.index(row[10])
    row[10] = mi

    dataset.append(row)

# Replace the missing values
for index,row in enumerate(dataset):
    dataset[index]=[value if value not in ["NR",""] else -1 for value in row]

# Filter the dataset based on the column name
feature_to_filter=["crm","annee_naissance","annee_permis", "kmage_annuel", "anc_veh", "var9", "marque"]
indexes_to_filter=[]
for feature in feature_to_filter:
    indexes_to_filter.append(header.index(feature))

dataset_filtered=[]
for row in dataset:
    dataset_filtered.append([float(row[index]) for index in indexes_to_filter])
# Build the structure containing the target
targets=[]
for row in dataset:
    targets.append(float(row[header.index("prime_tot_ttc")]))

#Split the datasets to have one for learning and the other for the test
train_dataset=[]
test_dataset=[]
train_target=[]
test_target=[]

for row,target in zip(dataset_filtered,targets):
    train_dataset.append(row)
    train_target.append(target)
    if random.random() < 0.70:
        pass
        #train_dataset.append(row)
        #train_target.append(target)
    else:
        test_dataset.append(row)
        test_target.append(target)

#Build the model

predicts = dict()

#for model in [ExtraTreesRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor()]:
for model in [RandomForestRegressor(), GradientBoostingRegressor()]:
    #model=ExtraTreesRegressor()
    #model=RandomForestRegressor()
    #model=GradientBoostingRegressor()
    #model=GaussianNB()
    #model=Ridge()
    #model=KNeighborsRegressor()
    #model=DecisionTreeRegressor()
    model.fit(train_dataset,train_target)

    #Predict with the model
    predictions=model.predict(test_dataset)

    ### Cross Validation ###

    #cv = StratifiedKFold(train_dataset, n_folds=5)

    ###scoring
    scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    ### getting the predictions ###
    #predicted = cross_validation.cross_val_predict(clf, train_dataset, train_target, cv=10)
    #print metrics.accuracy_score(train_target, predicted) 
    model.fit(train_dataset,train_target)
    predictions=model.predict(test_dataset)

    #Evaluate the quality of the prediction
    print sklearn.metrics.mean_absolute_error(predictions,test_target)

    filepath_train="./data/ech_test.csv"
    file_open=open(filepath_train,"r")
    # Open the csv reader over the file
    csv_reader=csv.reader(file_open,delimiter=";")
    # Read the first line which is the header
    header=csv_reader.next()
    # Load the dataset contained in the file
    dataset=[]
    for row in csv_reader:
        try:
            mi = marque.index(row[3])
        except:
            marque.append(row[3])
            mi = marque.index(row[3])
        row[3] = mi

        try:
            mi = job.index(row[10])
        except:
            job.append(row[10])
            mi = job.index(row[10])
        row[10] = mi

        dataset.append(row)

    # Replace the missing values
    for index,row in enumerate(dataset):
        dataset[index]=[value if value not in ["NR",""] else -1 for value in row]

    dataset_filtered=[]
    for row in dataset:
        dataset_filtered.append([float(row[index]) for index in indexes_to_filter])

    predictions=model.predict(dataset_filtered)

    for i in range(len(predictions)):
        predicts[str(i + 300001)] = predicts.get(str(i + 300001), 0)
        predicts[str(i + 300001)] += predictions[i]

f = open("result.csv", "w") 
f.write("ID;COTIS\n")

for i in xrange(300001, 330000 + 1):
    f.write(str(i) + ";" + str(predicts[i] / 2) + "\n")
f.close()