# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 10:16:38 2017

@author: evanc
"""

import pandas as pd

df = pd.read_csv("C:\\Users\\evanc\\Python\\Data Analytics Project\\data.csv")

def process_categorical():
    # currently categories are represented by integers and stored as such. 
    # We need to change them to categorical so we can get the dummies.
    global df
    
    categories = ["Gender",
    "Ethnicitiy",
    "Citizen_of_US",
    "Education2",
    "Language",
    "TestedForHIV",
    "EverSmokedWeed",
    "EverHardDrug",
    "EverUseCocaine",
    "EverInjectDrugNeedle",
    "BeenToRehab"
    ,"CoveredByHealthCare"
    ,'CoveredByPrivate'
    ,'CoveredByPublicHC'
    ,'EverHadHepB'
    ,'EverHadHepC'
    ,'GenealHealthCondition'
    ,'Fem12mosUnablePregnant'
    ,'FemEverHadReproductiveInfect'
    ,'FemEverBeenPregn'
    ,'FemCurrentlyPregnant'
    ,'BothEverHadSex'
    ,'MalesEverHadVaginalSex'
    ,'MalesEverOralSexWithFemale'
    ,'MalesEverHadAnalSexFemales'
    ,'MalesEverHadAnalSexMales'
    ,'FemalesEverHadSexWithMan'
    ,'FemalesEverHadOralSex'
    ,'FemalesEverHadAnalSex'
    ,'FemalesEverHadSexWithFemale'
    ,'FemalesEverHadHPV'
    ,'BothEverHadHerpes'
    ,'BothEverHadGenitalWarts'
    ,'BothEverHadGono'
    ,'BothEverHadChl'
    ,'SexualOrientation']
    
    for i in range(len(categories)):
        category = categories[i]
        df[category] = df[category].astype('category')
       
def train_and_target():
    global target
    global df
    global train
    df.set_index("SEQN", inplace = True)
    target = df.ChlamydiaResults
    df.drop("ChlamydiaResults", axis = 1, inplace = True)
    train = df
    train = pd.get_dummies(train)
    
def run_svm():
    weights = {0:67.42, 1:892.54}
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    sv = svm.SVC(C=3, class_weight = weights)
    clf = sv.fit(train,target)
    scores = cross_val_score(clf,train,target,cv=10)
    print(scores)

def run_decision_tree():
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    tree = tree.DecisionTreeClassifier(class_weight ='balanced')
    clf = tree.fit(train,target)
    scores = cross_val_score(clf,train,target,cv =10)
    print(scores)
    
if __name__ == "__main__":
    process_categorical()
    train_and_target()
    run_svm()
    run_decision_tree()
