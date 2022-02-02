

## Importing the libraries
import pandas as pd

#importing data set
dataset=pd.read_csv("breast_cancer.csv")
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values

#Spliting into train ,test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Traning Model
from sklearn.linear_model import LogisticRegression
classification=LogisticRegression(random_state=0)
classification.fit(X_train,Y_train)

#Predicting Result for test set
y_pred=classification.predict(X_test)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
print(cm)

#Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classification, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Predicting Result for Custom input
Clump_Thickness = int(input("\n\t Enter the Clump_Thickness  : "))
Uniformity_of_Cell_Size = int(input("\n\t Enter the Uniformity of Cell Size : "))
Uniformity_of_Cell_Shape = int(input("\n\t Enter the Uniformity of Cell Shape : "))
Marginal_Adhesion = int(input("\n\t Enter the value of Marginal Adhesion : "))
Single_Epithelial_Cell_Size = int(input("\n\t Enter the Single Epithelial Cell Size : "))
Bare_Nuclei = int(input("\n\t Enter the value of Bare Nuclei : "))
Bland_Chromatin = int(input("\n\t Enter the value of Bland Chromatin : "))
Normal_Nucleoli = int(input("\n\t Enter the value of Normal Nucleoli : "))
Mitoses = int(input("\n\t Enter the value of Mitoses Mitoses : "))
inp_arr = [Clump_Thickness,Uniformity_of_Cell_Size,Uniformity_of_Cell_Shape,Marginal_Adhesion , Single_Epithelial_Cell_Size,Bare_Nuclei,Bland_Chromatin,Normal_Nucleoli,Mitoses]
print("\n\t The values you entered : ", inp_arr)

out=classification.predict([inp_arr])
# M = malignant, B = benign)
if(out[0]==2):
    print("\n\t The brest cancer tumor is at benign stage")
else:
    print("\n\t The brest cancer tumor is at malignant stage")
