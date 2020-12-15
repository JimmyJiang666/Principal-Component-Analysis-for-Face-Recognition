import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Download data set
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 60)
# print(faces.target_names,"\n",faces.images.shape)

# look at a few picture and labels in the data set
# fig, ax = plt.subplots(4,5, figsize=(8,8))
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap = 'bone')
#     axi.set(xticks=[], yticks=[], xlabel = faces.target_names[faces.target[i]])
# plt.savefig("test.png")

# Split data set into training and test sets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target)

from sklearn.svm import SVC
from sklearn.decomposition import PCA

k = 125 # after experiments, 
pca = PCA(n_components = k)
pca.fit(Xtrain)
projected_Xtrain = pca.transform(Xtrain)# projected_Xtrain is the projected training data
sigma = pca.singular_values_ #we use this to scale our transformed Xtrain


def scale(myArray): # be cautious this scaling function will change the original array
    for item in myArray:
        for i in range(k):
            item[i] = (1/(sigma[i]))*item[i]
    return myArray

scaled_projected_Xtrain = scale(projected_Xtrain)


# training
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C = 1E3) #hard margin

projected_Xtest = pca.transform(Xtest)


##########################################################
# with scaling turned on
scaled_projected_Xtest = scale(projected_Xtest)
svc.fit(scaled_projected_Xtrain, Ytrain)
predict_Ytest = svc.predict(scaled_projected_Xtest)
#########################################################

###################################################
# to test no scaling effect
# svc.fit(projected_Xtrain, Ytrain)
# predict_Ytest = svc.predict(projected_Xtest)
###################################################

def test_accuracy(predict, actual):# test performance by accuracy
	cnt = 0
	for i in range(len(actual)):
		if actual[i]!=predict[i]:
			cnt += 1
	return (len(actual) - cnt)/(len(actual))

print("with k = ", k, " and scaling on, the accuracy is :", test_accuracy(predict_Ytest, Ytest))
