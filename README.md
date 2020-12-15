# Principal-Component-Analysis-for-Face-Recognition
Using PCA technique to train and test a model to handle face recognition tasks.

In Face Recognition.ipynb (attached), we download a set of labelled images. Each image
is the face of a person labelled with the name of that person. faces.images is an array of
images each having 62 × 47 pixels. The names are stored in the array faces.target names.
faces.target[i] gives the index of the name of the person in faces.images[i] in the array
faces.target names. In other words, the name of the person shown in faces.images[i] is
faces.target names[faces.target[i]].

The goal is to use a part of the data set for learning a classifier and test its performance on
the remaining data. We use the function train test split to split the data set. We think of
each image as a vector in 62×47 = 2914 dimensions. The first step is to reduce the dimension
of the data set using PCA and then use kernel SVM for classification in the reduced dimension.
For PCA, the data matrix to work with is Xtrain. Project each data point onto the subspace
spanned by the first k principal components where k is relatively small (e.g. 100). Recall that
the variance of the data set along the i^th principal component is σ_2^i, σi being the i
th singular value. Since SVM uses distances, we would like all dimensions to have the same “scale”. To
this end, scale the coordinates of the projected data along ith principal component vector by 1 σ_i
so that the variance along each dimension is the same ( it is 1/N where N is the number
of data points).

Finally train a support vector classifier with rbf kernel using the projected and scaled data
points and their corresponding labels in Ytrain.

Test the performance of the classifier on the test set Xtest and compute the percentage of
images correctly classified by comparing with the correct labels in Ytest. You can vary the
value of k and see how it affects the performance of the algorithm.
Repeat the experiment again, this time turning off the scaling along the i th principal
component on the projected data. How is the performance on the test set this time?
