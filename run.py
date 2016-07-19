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
import copy

#Set the seed
random.seed(42)

job = []

company = dict()
company_count = dict()
job2 = dict()
job2_count = dict()
veh = dict()
veh_count = dict()

morque = ["SOVRA", "SUNBEAM", "DETHLEFFS", "MARUTI", "AMC", "LORENZ", "SALMSON", "JIDE", "TEILHOL", "CHEYENNE", "WIESMANN", "CATERHAM", "LOHR", "MVS", "HOTCHKISS", "UNIMOG", "AUSTIN-HEALEY", "TRIUMPH", "MOKE", "DONKERVOORT", "PANTHER", "NR", "TVR", "WILLYS", "LOTUS", "PANHARD", "EBS", "EAGLE", "INTERNATIONAL", "GRANDIN", "RAMBLER", "ENGIN LOISIRS", "AUTO-UNION", "SATURN", "BREMACH", "BUICK", "NSU", "PIAGGIO", "STEYR PUCH", "DAF", "HUMMER", "PININFARINA", "MARTIN", "AUTOBIANCHI", "VEH SPECIAL", "ALPINE RENAULT", "LAFER", "MEGA", "CADILLAC", "GME", "BENTLEY", "AUTOLAND", "MORGAN", "TALBOT", "VAUXHALL", "SECMA", "SMS", "BRM", "RELIANT", "PLYMOUTH", "DE LOREAN", "ACMA", "AUSTIN", "ENGIN TRAVAUX PUBLIC", "PGO", "SIATA", "ASTON MARTIN", "MORRIS", "NECKAR", "MAHINDRA", "PORSCHE", "TATA", "PONCIN", "CHENARD ET WALKER", "DAIHATSU", "MASERATI", "ASIA", "UMM", "MCC", "JAGUAR", "JEEP", "FERRARI", "STRAUBENHARDT", "CARBODIES", "CHEVROLET", "DELAHAYE", "DATSUN", "DANGEL", "AMPHICAR", "SAAB", "SANTANA", "SINGER", "MG", "LADA", "LANCIA", "RAYTON FISSORE", "YAMAHA", "ROLLS ROYCE", "BEDFORD", "LINCOLN", "SAVIEM", "HYUNDAI", "BMW", "MAZDA", "SMART", "EBRO", "AUVERLAND", "ARO", "DODGE", "NISSAN", "SIMCA", "PONTIAC", "ISUZU", "GMC", "KIA", "MITSUBISHI", "COURNIL", "FORD", "LAMBORGHINI", "ROVER", "LAND-ROVER", "BERTONE", "INFINITI", "ENGIN SPECIAL", "AUDI", "RENAULT", "LEXUS", "MATRA", "SUBARU", "CITROEN", "MINI", "FIAT", "SSANGYONG", "HANOMAG", "MERCURY", "CHRYSLER", "SUZUKI", "MERCEDES", "VOLVO", "IVECO", "TOYOTA", "DAIMLER", "PEUGEOT", "VOLKSWAGEN", "OPEL", "SKODA", "DAEWOO", "HONDA", "DACIA", "ALFA ROMEO", "OLDSMOBILE", "FSO", "SEAT", "PROTON", "LDV"]

job3 = ["agriculteur exploitant", "homme ou femme au foyer", "commercant", "chef d'entreprise", "profession liberale", "artisan", "cadre, ingenieur", "contremaitre, agent de maitrise", "demandeur d'emploi", "profession intermediaire de la sante et du travail social", "technicien", "profession de l'information des arts et des spectacles", "agent, employe", "enseignant, formateur, chercheur", "educateur, animateur, moniteur", "etudiant", "ouvrier"]

anc_veh = ["36", "19", "40", "30", "17", "39", "16", "21", "20", "27", "22", "31", "41", "32", "35", "23", "29", "26", "33", "28", "11", "34", "99", "37", "42", "18", "10", "24", "15", "13", "12", "9", "25", "4", "38", "8", "0", "14", "1", "2", "5", "3", "6", "7"]

dataset=[]
row_dataset=[]

# Open the file
for filepath_train in ["./data/ech_apprentissage.csv"]:#, "./added2.csv"]:
    #filepath_train="./data/ech_apprentissage.csv"
    file_open=open(filepath_train,"r")
    # Open the csv reader over the file
    csv_reader=csv.reader(file_open,delimiter=";")
    # Read the first line which is the header
    header=csv_reader.next()
    # Load the dataset contained in the file

    for row in csv_reader:
        company[row[3]] = company.get(row[3], 0)
        company[row[3]] += float(row[header.index("prime_tot_ttc")])

        company_count[row[3]] = company_count.get(row[3], 0)
        company_count[row[3]] += 1

        if row[3] in morque:
            mi = morque.index(row[3])
        else:
            mi = -1

        row[3] = mi

        job2[row[10]] = job2.get(row[10], 0)
        job2[row[10]] += float(row[header.index("prime_tot_ttc")])

        job2_count[row[10]] = job2_count.get(row[10], 0)
        job2_count[row[10]] += 1

        veh[row[5]] = veh.get(row[5], 0)
        veh[row[5]] += float(row[header.index("prime_tot_ttc")])

        veh_count[row[5]] = veh_count.get(row[5], 0)
        veh_count[row[5]] += 1

        if row[10] in job3:
            mi = job3.index(row[10])
        else:
            mi = -1

        row[10] = mi

        #if row[5] in anc_veh:
        #    mi = anc_veh.index(row[5])
        #else:
        #    mi = -1
    #
    #    row[5] = mi

        dataset.append(row)

# Replace the missing values
for index,row in enumerate(dataset):
    dataset[index]=[value if value not in ["NR",""] else -1 for value in row]

# Filter the dataset based on the column name
feature_to_filter = ["crm", "annee_naissance", "annee_naissance", "annee_permis", "kmage_annuel", "anc_veh", "var9", "marque"] \
        + ["var1", "var2", "var3", "var4", "var5", "var7", "var10", "var11", "var12", "var13", "var15", "var16", "var17", "var18", "var19", "var20", "var21", "var22"]

indexes_to_filter=[]
#indexes_to_filter2=[]
for feature in feature_to_filter:
    indexes_to_filter.append(header.index(feature))
#for feature in feature_to_filter2:
#    indexes_to_filter2.append(header.index(feature))

dataset_filtered=[]
#dataset_filtered2=[]
for row in dataset:
    dataset_filtered.append([float(row[index]) for index in indexes_to_filter])
    #dataset_filtered2.append([float(row[index]) for index in indexes_to_filter2])

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

filepath_train="./data/ech_test.csv"
file_open=open(filepath_train,"r")
csv_reader=csv.reader(file_open,delimiter=";")
header=csv_reader.next()
dataset=[]
for row in csv_reader:
    row_dataset.append(copy.deepcopy(row))

    if row[3] in morque:
        mi = morque.index(row[3])
    else:
        mi = -1
    row[3] = mi

    if row[10] in job3:
        mi = job3.index(row[10])
    else:
        mi = -1
    row[10] = mi

    #if row[5] in anc_veh:
    #    mi = anc_veh.index(row[5])
    #else:
    #    mi = -1

    #row[5] = mi

    dataset.append(row)

# Replace the missing values
for index,row in enumerate(dataset):
    dataset[index]=[value if value not in ["NR",""] else -1 for value in row]

dataset_filtered=[]
for row in dataset:
    dataset_filtered.append([float(row[index]) for index in indexes_to_filter])

#for model in [ExtraTreesRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor()]:
#for model in [ExtraTreesRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]:
for model in [RandomForestRegressor(), GradientBoostingRegressor(max_depth = 9)]:
    model.fit(train_dataset,train_target)

    #Predict with the model
    predictions=model.predict(test_dataset)

    ###scoring
    scores = cross_validation.cross_val_score(model, train_dataset, train_target, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    ### getting the predictions ###
    print sklearn.metrics.mean_absolute_error(predictions,test_target)

    predictions=model.predict(dataset_filtered)
    for i in range(len(predictions)):
        predicts[str(i + 300001)] = predicts.get(str(i + 300001), 0)
        predicts[str(i + 300001)] += predictions[i]

f = open("result.csv", "w") 
f.write("ID;COTIS\n")

#print len(predicts.keys())
#print predicts.keys()
for i in xrange(300001, 330000 + 1):
    f.write(str(i) + ";" + str(predicts[str(i)] / 2) + "\n")
f.close()

f = open("added1111.csv", "w") 
dataset_filtered=[]
print len(row_dataset)
for i, row in enumerate(row_dataset):
    row = [str(elem) for elem in row]
    f.write(";".join(row)+";"+str(predicts[str(i + 300001)] / 2)+"\n")
f.close()

#for k in company.keys():
#    company[k] /= company_count[k]
#
#print company
#
#for k in job2.keys():
#    job2[k] /= job2_count[k]
#
#print job2
#
#for k in veh.keys():
#    veh[k] /= veh_count[k]
#
#print veh