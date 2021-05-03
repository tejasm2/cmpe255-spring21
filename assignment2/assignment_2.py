from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images.shape)
    return faces

faces = load_data()
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=42)


#Initializing GridSearchCV
parameters = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid_cv = GridSearchCV(model, parameters)
grid_cv.fit(X_train, y_train)


#Printing the best parameters found using GridSearchCV
print(grid_cv.best_params_)


#Making Predictions
model = grid_cv.best_estimator_
pred = model.predict(X_test)


#Getting the classification report (Precision,recall,f1-score,support)
print(classification_report(y_test, pred, target_names=faces.target_names))


#Plotting the predictions
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1], color='black' if pred[i] == y_test[i] else 'red')
fig.suptitle('Predicted Labels in Black | Incorrect Labels in Red', size=14)
plt.show()


#Plotting the Confusion Matrix
c_matrix = confusion_matrix(y_test, pred)
sns.heatmap(c_matrix.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()