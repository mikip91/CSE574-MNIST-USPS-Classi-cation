
# coding: utf-8

# ## Load MNIST 

# In[160]:


import pickle
import gzip
import numpy as np
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
from collections import Counter


# In[161]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
print(training_data[0].shape, training_data[1].shape)
print(validation_data[0].shape, validation_data[1].shape)
print(test_data[0].shape, test_data[1].shape)
final_data = np.r_[training_data[0], validation_data[0]]
final_data_target= np.append(training_data[1], validation_data[1])
print(final_data.shape, final_data_target.shape)
f.close()


# In[111]:


majority_voting_list = []


# ## Implementation 1: Softmax Regression Implementation from scratch

# In[4]:


def softmax(Z, epsilon=1e-9):
    e = np.exp(Z - np.max(Z))
    if e.ndim == 1:
        return e / np.sum(e, axis=0) + epsilon
    else:  
        return e / np.array([np.sum(e, axis=1)]).T + epsilon


# In[5]:


def infer(W, X):
    X_ones = np.hstack((X, np.ones(((X.shape[0]), 1))))
    XW = np.dot(X_ones, W)
    smax = softmax(XW)
    return smax


# In[6]:


eta = 1e-2
def one_hot_encode(labels_list, max_number):
    samples_number = len(labels_list)
    b = np.zeros((samples_number, max_number))
    b[np.arange(samples_number), labels_list] = 1
    return b


# In[7]:


def loss(W, X, Y):
    m = X.shape[0]
    Y_tilde = infer(W, X)    
    return (-1 / m) * np.sum(np.log(Y_tilde) * Y) + eta / 2 * np.sum(W * W)


# In[8]:


num_classes = 10
y_onehot = one_hot_encode(final_data_target, num_classes)


# In[9]:


def get_grad(W, X, Y):   
    X_alt = np.hstack((X, np.ones(((X.shape[0]), 1))))
    m = X.shape[0]
    Y_tilde = infer(W, X)   
    return (-1 / m) * np.dot(X_alt.T, (Y - Y_tilde)) + eta * W


# In[10]:


n_classes=10
def train(X_train, y_train, batch_size=512, num_epoch=256, n_classes=n_classes, step=1e-3, plot_loss=True):
    losses = []
    n_features = X_train.shape[1]
    w = np.random.randn(n_features+1, n_classes)/n_features
    for epoch in range(num_epoch):        
        for iter_num, (x_batch, y_batch) in enumerate(zip(np.split(X_train, batch_size), np.split(y_train, batch_size))):
            grad = get_grad(w, x_batch, one_hot_encode(y_batch, n_classes))
            gradient_step = step * grad
            w -= gradient_step
            losses.append(loss(w, x_batch, one_hot_encode(y_batch, n_classes)))
            
    if plot_loss:
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("epochs")
        plt.show()    
    return w


# ## Prediction of Logistic Regression

# In[11]:


def make_prediction(X, W):
    probability_matrix = infer(W, X)
    return np.array([np.argmax(t) for t in probability_matrix])


# In[12]:


W = train(final_data, final_data_target, num_epoch=1000)
print(W.shape)


# In[112]:


right=0
wrong=0
y_pred = make_prediction(test_data[0], W)

majority_voting_list.append(y_pred)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
for i,j in zip(y_pred,test_data[1]): 
    if j == i:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Implementation 1:Testing Accuracy for Softmax Regression with MNIST data: " + str(right/(right+wrong)*100))
print("Implementation 1:Confusion Matrix for Softmax Regression with MNIST data:")
print(confusion_matrix(test_data[1], (y_pred)))


# ## Implementation 2: Softmax Regression Using Solver as `Ibfgs`

# In[113]:


mul_lr = linear_model.LogisticRegression(multi_class='multinomial',solver ='lbfgs').fit(final_data, final_data_target)


# In[114]:


y_pred = accuracy_score(test_data[1], mul_lr.predict(test_data[0]))
confusion_matrix_logisticReg2 = confusion_matrix(test_data[1], mul_lr.predict(test_data[0]))
majority_voting_list.append(mul_lr.predict(test_data[0]))
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
print("Implementation 2:Testing Accuracy for Softmax Regression with MNIST data: "+ str(y_pred))
print("Implementation 2:Confusion Matrix for Softmax Regression with MNIST data:")
print(confusion_matrix_logisticReg2)


# ## Implementation 3: Softmax Regression Using Solver as `Ibfgs` , Penalty as `l2` and `warm_start` as true 

# In[14]:


mul_lr2 = linear_model.LogisticRegression(multi_class='multinomial',solver ='lbfgs',penalty ='l2',C=1e5, warm_start=True).fit(final_data, final_data_target)


# In[115]:


y_pred = accuracy_score(test_data[1], mul_lr2.predict(test_data[0]))
confusion_matrix_logisticReg3 = confusion_matrix(test_data[1], mul_lr2.predict(test_data[0]))
majority_voting_list.append(mul_lr2.predict(test_data[0]))
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
print("Implementation 3:Testing Accuracy for softmax regression with MNIST data: "+str(y_pred))
print("Implementation 3:Confusion Matrix for Softmax Regression with MNIST data:")
print(confusion_matrix_logisticReg3)


# ## Implementation 4: Neural Network Using Keras

# In[24]:


(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
num_classes=10
image_vector_size=28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
NN_model = Sequential()
NN_model.add(Dense(units=32, activation='sigmoid', input_shape=(784,)))
NN_model.add(Dense(units=num_classes, activation='softmax'))
NN_model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = NN_model.fit(x_train, y_train, batch_size=128, epochs=256,verbose=1,validation_split=.1)


# In[116]:


(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

NN_model_pred = NN_model.predict(np.array(x_test))
# confusion_matrix_NN1 = confusion_matrix(test_data[1],NN_model_pred)
NN_predicted_value = NN_model_pred.argmax(axis=-1)
confusion_matrix_NN1 = confusion_matrix(NN_predicted_value,test_data[1])
loss,accuracy = NN_model.evaluate(x_test, y_test, verbose=False)
majority_voting_list.append(NN_predicted_value)
print("Majority List: " + str(np.matrix(majority_voting_list).shape))
print("Implementation 4: Testing Accuracy for Neural Network Using MNIST Dataset:" +str(accuracy))
print("Implementation 4: Confusion Matrix for Neural Network Using MNIST Dataset:")
print(confusion_matrix_NN1)


# ## Implementation 5: Neural Network Using  Library

# In[25]:


NN_Classifier2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=256, random_state=1)
NN_Classifier2.fit(final_data, final_data_target)  


# In[117]:


NN_pred=NN_Classifier2.predict(test_data[0])
y_pred = accuracy_score(test_data[1],NN_pred)
majority_voting_list.append(NN_pred)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_NN2 = confusion_matrix(test_data[1],NN_pred)
print("Implementation 5: Testing Accuracy for Neural Network Using MNIST Dataset:"+str(y_pred))
print("Implementation 5: Confusion Matrix for Neural Network Using MNIST Dataset:")
print(confusion_matrix_NN2)


# ## Implementation 6: Neural Network Using Library

# In[26]:


NN_Classifier3 = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=200, random_state=1, max_iter=50)
NN_Classifier3.fit(final_data, final_data_target)                         


# In[118]:


NN_pred2=NN_Classifier3.predict(test_data[0])
y_pred = accuracy_score(test_data[1],NN_pred)
majority_voting_list.append(NN_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_NN3 = confusion_matrix(test_data[1],NN_pred2)
print("Implementation 6: Testing Accuracy for Neural Network Using MNIST Dataset:"+str(y_pred))
print("Implementation 6: Confusion Matrix for Neural Network Using MNIST Dataset:")
print(confusion_matrix_NN3)


# ## Implementation 7: SVM Using kernel as 'linear'

# In[82]:


X_train, y_train = final_data, final_data_target
X_test, y_test = test_data[0], test_data[1]
SVM_classifier1 = SVC(kernel='linear', C=2, gamma= 0.05);
SVM_classifier1.fit(X_train, y_train)


# In[123]:


X_test, y_test = test_data[0], test_data[1]
SVM_pred1=SVM_classifier1.predict(X_test)
y_pred = accuracy_score(test_data[1],SVM_pred1)
majority_voting_list.append(SVM_pred1)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_SVM1 = confusion_matrix(test_data[1],SVM_pred1)
print("Implementation 7: Testing Accuracy for SVM Using MNIST Dataset:"+str(y_pred))
print("Implementation 7: Confusion Matrix for SVM Using MNIST Dataset:")
print(confusion_matrix_SVM1)


# ## Implementation 8: SVM Using radial basis function and gamma =1

# In[ ]:


X_train, y_train = final_data, final_data_target
SVM_classifier2 = SVC(kernel='rbf', gamma= 1, max_iter= 30);
SVM_classifier2.fit(X_train, y_train)


# In[ ]:


X_test, y_test = test_data[0], test_data[1]
SVM_pred2=SVM_classifier2.predict(X_test)
y_pred = accuracy_score(test_data[1],SVM_pred2)
majority_voting_list.append(SVM_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_SVM2 = confusion_matrix(test_data[1],SVM_pred2)
print("Implementation 8: Testing Accuracy for SVM Using MNIST Dataset:"+str(y_pred))
print("Implementation 8: Confusion Matrix for SVM Using MNIST Dataset:")
print(confusion_matrix_SVM2)


# ## Implementation 9: SVM Using radial basis function and gamma as default

# In[107]:


X_train, y_train = final_data, final_data_target
SVM_classifier3 = SVC(kernel='rbf', max_iter= 30);
SVM_classifier3.fit(X_train, y_train)


# In[109]:


X_test, y_test = test_data[0], test_data[1]
SVM_pred3=SVM_classifier3.predict(X_test)
y_pred = accuracy_score(test_data[1],SVM_pred3)
majority_voting_list.append(SVM_pred3)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_SVM3 = confusion_matrix(test_data[1],SVM_pred3)
print("Implementation 9: Testing Accuracy for SVM Using MNIST Dataset:"+str(y_pred))
print("Implementation 9: Confusion Matrix for SVM Using MNIST Dataset:")
print(confusion_matrix_NN3)


# ## Implementation 10: SVM Using kernel as `poly`

# In[108]:


X_train, y_train = final_data, final_data_target
SVM_classifier4 = SVC(kernel='poly', C=5, max_iter=20);
SVM_classifier4.fit(X_train, y_train)


# In[110]:


X_test, y_test = test_data[0], test_data[1]
SVM_pred4=SVM_classifier4.predict(X_test)
y_pred = accuracy_score(test_data[1],SVM_pred4)
majority_voting_list.append(SVM_pred4)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_SVM4 = confusion_matrix(test_data[1],SVM_pred4)
print("Implementation 10: Testing Accuracy for SVM Using MNIST Dataset:"+str(y_pred))
print("Implementation 10: Confusion Matrix for SVM Using MNIST Dataset:")
print(confusion_matrix_NN4)


# ## Implementation 11: Random Forest with n_estimators as 10

# In[39]:


X_train, y_train = final_data, final_data_target
RF_classifier1 = RandomForestClassifier(n_estimators=10);
RF_classifier1.fit(X_train, y_train)


# In[119]:


X_test, y_test = test_data[0], test_data[1]
RF_pred1=RF_classifier1.predict(X_test)
y_pred = accuracy_score(test_data[1],RF_pred1)
majority_voting_list.append(RF_pred1)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_RF1 = confusion_matrix(test_data[1],RF_pred1)
print("Implementation 11: Testing Accuracy for Random Forest Using MNIST Dataset:"+str(y_pred))
print("Implementation 11: Confusion Matrix for Random Forest Using MNIST Dataset:")
print(confusion_matrix_RF1)


# ## Implementation 12: Random Forest with n_estimators as 100

# In[41]:


X_train, y_train = final_data, final_data_target
RF_classifier2 = RandomForestClassifier(n_estimators=100, criterion= 'entropy');
RF_classifier2.fit(X_train, y_train)


# In[120]:


X_test, y_test = test_data[0], test_data[1]
RF_pred2=RF_classifier2.predict(X_test)
y_pred = accuracy_score(test_data[1],RF_pred2)
majority_voting_list.append(RF_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_RF2 = confusion_matrix(test_data[1],RF_pred2)
print("Implementation 12: Testing Accuracy for Random Forest Using MNIST Dataset:"+str(y_pred))
print("Implementation 12: Confusion Matrix for Random Forest Using MNIST Dataset:")
print(confusion_matrix_RF2)


# ## Implementation 13: Random Forest with n_estimators as 200

# In[43]:


X_train, y_train = final_data, final_data_target
RF_classifier3 = RandomForestClassifier(n_estimators=200);
RF_classifier3.fit(X_train, y_train)


# In[121]:


X_test, y_test = test_data[0], test_data[1]
RF_pred3=RF_classifier3.predict(X_test)
y_pred = accuracy_score(test_data[1],RF_pred3)
majority_voting_list.append(RF_pred3)
print("Majority List: " +  str(np.matrix(majority_voting_list).shape))
confusion_matrix_RF3 = confusion_matrix(test_data[1],RF_pred3)
print("Implementation 13: Testing Accuracy for Random Forest Using MNIST Dataset:"+str(y_pred))
print("Implementation 13: Confusion Matrix for Random Forest Using MNIST Dataset:")
print(confusion_matrix_RF3)


# ## Implementation 14: CNN Implementation

# In[72]:


from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
img_x, img_y = 28, 28
batch_size = 128
num_classes = 10
epochs = 10
x_train = x_train.reshape(final_data.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(test_data[0].shape[0], img_x, img_y, 1)

y_train = keras.utils.to_categorical(final_data_target, num_classes)
y_test = keras.utils.to_categorical(test_data[1], num_classes)

input_shape = (img_x, img_y, 1)

CNN_model = Sequential()
CNN_model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
CNN_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
CNN_model.add(Conv2D(64, (5, 5), activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))
CNN_model.add(Flatten())
CNN_model.add(Dense(1000, activation='relu'))
CNN_model.add(Dense(num_classes, activation='softmax'))

CNN_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
CNN_model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[history])


# In[122]:


(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
x_train = x_train.reshape(final_data.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(test_data[0].shape[0], img_x, img_y, 1)

y_train = keras.utils.to_categorical(final_data_target, num_classes)
y_test = keras.utils.to_categorical(test_data[1], num_classes)

score = CNN_model.evaluate(x_test, y_test, verbose=0)
CNN_model_pred = CNN_model.predict(np.array(x_test))
CNN_predicted_value = CNN_model_pred.argmax(axis=-1)
confusion_matrix_CNN1 = confusion_matrix(CNN_predicted_value,test_data[1])
loss,accuracy = CNN_model.evaluate(x_test, y_test, verbose=False)
majority_voting_list.append(CNN_predicted_value)
print(CNN_predicted_value)
print("Majority List: " + str(np.matrix(majority_voting_list).shape))
print('Implementation 14: Testing Accuracy for CNN Using MNIST Dataset:', score[1])
print("Implementation 14: Confusion Matrix for CNN Using MNIST Dataset:")
print(confusion_matrix_CNN1)


# ## Majority Voting for MNIST Dataset

# In[124]:


final_majority_list = np.matrix(majority_voting_list)
final_majority_list = mode(final_majority_list)
# for i in range(len(test_data[1])):
#     most_common, num_most_common = Counter(final_majority_list.flat).most_common(1)[0]
#     ensemble_list.append(most_common)
# print("Majority Voting" +ensemble_list)

print("Final Majority List: " +str((final_majority_list[0][0]).shape))
y_pred = accuracy_score(test_data[1],final_majority_list[0][0])
MV_confusion_matrix = confusion_matrix(test_data[1],final_majority_list[0][0])
# print("Final Majority List: " +str(ensemble_list.shape))
# y_pred = accuracy_score(test_data[1], ensemble_list)
# MV_confusion_matrix = confusion_matrix(test_data[1],ensemble_list)
print('Testing Accuracy for Majority Voting Using MNIST Dataset:', str(y_pred))
print("Confusion Matrix for Majority Voting  Using MNIST Dataset:")
print(MV_confusion_matrix)


# ## Bagging using Softmax Regression for MNIST Data

# In[165]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
bagging1 = BaggingClassifier(base_estimator=mul_lr2, n_estimators=5, max_samples=0.8, max_features=0.8)
bagging1.fit(x_train, y_train)


# In[166]:


scores = cross_val_score(bagging1, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],bagging1.predict(x_test))
print(MV_confusion_matrix) 


# ## Voting Classifier

# In[187]:


from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
estimators = []
model1 = mul_lr
estimators.append(('Softmax Regression', model1))
model2 = mul_lr2
estimators.append(('Softmax Regression2', model2))
model3 = NN_Classifier2
estimators.append(('Neural Network Classifier', model3))
model4 = NN_Classifier3
estimators.append(('Neural Network Classifier2', model4))
model5 = SVM_classifier1
estimators.append(('SVM_Classifier1', model5))
model6 = RF_classifier1
estimators.append(('Random Forest', model6))
model7 = RF_classifier2
estimators.append(('Random Forest2', model7))
model8 = RF_classifier3
estimators.append(('Random Forest3', model8))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble,x_test, y_test, cv=5)
print(results.mean())


# ## Bagging using neural network for MNIST Data

# In[ ]:


bagging2 = BaggingClassifier(base_estimator=NN_Classifier2, n_estimators=5, max_samples=0.8, max_features=0.8)
bagging2.fit(x_train, y_train)


# In[ ]:


scores = cross_val_score(bagging2, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],bagging2.predict(x_test))
print(MV_confusion_matrix)


# ## Bagging using SVM for MNIST Data

# In[ ]:


bagging3 = BaggingClassifier(base_estimator=SVM_classifier1, n_estimators=5, max_samples=0.8, max_features=0.8)
bagging3.fit(x_train, y_train)


# In[ ]:


scores = cross_val_score(bagging3, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],bagging3.predict(x_test))
print(MV_confusion_matrix)


# ## Bagging using Random Forest for MNIST Data

# In[ ]:


bagging4 = BaggingClassifier(base_estimator=RF_classifier3, n_estimators=5, max_samples=0.8, max_features=0.8)
bagging4.fit(x_train, y_train)


# In[ ]:


scores = cross_val_score(bagging4, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],bagging4.predict(x_test))
print(MV_confusion_matrix)


# ## Boosting using Softmax Regression for MNIST Data

# In[170]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
boosting1 = AdaBoostClassifier(base_estimator=mul_lr2, n_estimators=10)
boosting1.fit(x_train, y_train)


# In[171]:


scores = cross_val_score(boosting1, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],boosting1.predict(x_test))
print(MV_confusion_matrix) 


# ## Boosting using Random Forest for MNIST Data

# In[172]:


(x_train, y_train)= (final_data, final_data_target)
(x_test, y_test) = (test_data[0], test_data[1])
boosting2 = AdaBoostClassifier(base_estimator=RF_classifier3, n_estimators=10)
boosting2.fit(x_train, y_train)


# In[173]:


scores = cross_val_score(boosting2, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(test_data[1],boosting2.predict(x_test))
print(MV_confusion_matrix) 


# ## Load USPS 

# In[19]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPS/Numerals'
savedImg = []
for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
USPSNumeralsTarMat = np.matrix(USPSTar).T
USPSNumeralsInputMat = np.matrix(USPSMat)
print(USPSNumeralsTarMat.shape)
print(USPSNumeralsInputMat.shape)


# In[20]:


USPSTestMat  = []
USPSTestTar  = []
curPath  = 'USPS/Test'
savedImg = []

curFolderPath = curPath + '/'
imgs =  os.listdir(curFolderPath)
for img in imgs:
    curImg = curFolderPath + '/' + img
    if curImg[-3:] == 'png':
        img = Image.open(curImg,'r')
        img = img.resize((28, 28))
        savedImg = img
        imgdata = (255-np.array(img.getdata()))/255
        USPSTestMat.append(imgdata)
        USPSTestTar.append(j) 
USPSTestTar = np.matrix(USPSTestTar).T
USPSInputMat = np.matrix(USPSTestMat)
print(USPSTestTar.shape)
print(USPSInputMat.shape)


# In[133]:


majority_voting_usps = []


# ## Implementation 1: Logistic Regression  using USPS

# In[134]:


right=0
wrong=0
y_pred = make_prediction(np.array(USPSNumeralsInputMat), W)

majority_voting_usps.append(y_pred)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
for i,j in zip(y_pred,USPSNumeralsTarMat): 
    if j == i:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Implementation 1: Testing Accuracy for Softmax Regression Using USPS Dataset: " + str(right/(right+wrong)*100))
print("Implementation 1: Confusion Matrix for Softmax Regression Using USPS Dataset:")
print(confusion_matrix(USPSNumeralsTarMat, (y_pred)))


# ## Implementation 2 : Logistic Regression  using USPS

# In[135]:


y_pred = accuracy_score(USPSNumeralsTarMat, mul_lr.predict(USPSNumeralsInputMat))
confusion_matrix_logisticReg2 = confusion_matrix(USPSNumeralsTarMat, mul_lr.predict(USPSNumeralsInputMat))
majority_voting_usps.append(mul_lr.predict(USPSNumeralsInputMat))
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
print('Implementation 2: Testing Accuracy for Softmax Regression Using USPS Dataset:'+str(y_pred))
print("Implementation 2: Confusion Matrix for Softmax Regression Using USPS Dataset:")
print(confusion_matrix_logisticReg2)


# ## Implementation 3 : Logistic Regression  using USPS

# In[136]:


y_pred = accuracy_score(USPSNumeralsTarMat, mul_lr2.predict(USPSNumeralsInputMat))
confusion_matrix_logisticReg3 = confusion_matrix(USPSNumeralsTarMat, mul_lr2.predict(USPSNumeralsInputMat))
majority_voting_usps.append(mul_lr2.predict(USPSNumeralsInputMat))
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
print('Implementation 3: Testing Accuracy for Softmax Regression Using USPS Dataset:'+str(y_pred))
print("Implementation 3: Confusion Matrix for Softmax Regression Using USPS Dataset:")
print(confusion_matrix_logisticReg3)


# ## Implementation 4: Neural Network using USPS

# In[137]:


(x_test, y_test) = (np.array(USPSNumeralsInputMat),USPSNumeralsTarMat)
x_test = x_test.reshape(x_test.shape[0], image_vector_size )
y_test = keras.utils.to_categorical(y_test, num_classes)
NN_model_pred = NN_model.predict(np.array(x_test))
print(NN_model_pred.shape)
NN_predicted_value2 = NN_model_pred.argmax(axis=-1)
print(NN_predicted_value2.shape)
confusion_matrix_NN4 = confusion_matrix(NN_predicted_value2,USPSNumeralsTarMat)
loss,accuracy = NN_model.evaluate(x_test, y_test, verbose=False)
majority_voting_list.append(NN_predicted_value2)
print("Majority List: " + str(np.matrix(majority_voting_list).shape))
print("Implementation 4: Testing Accuracy for Neural Network Using USPS Dataset:" +str(accuracy))
print("Implementation 4: Confusion Matrix for Neural Network Using USPS Dataset:")
print(confusion_matrix_NN4)


# ## Implementation 5 & 6: Neural Network using USPS 

# In[138]:


NN_pred= NN_Classifier2.predict(USPSNumeralsInputMat)
y_pred = accuracy_score(USPSNumeralsTarMat,NN_pred)
majority_voting_usps.append(NN_pred)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_NN2 = confusion_matrix(USPSNumeralsTarMat,NN_pred)
print('Implementation 5: Testing Accuracy for Neural Network Using USPS Dataset:'+str(y_pred))
print("Implementation 5: Confusion Matrix for Neural Network Using USPS Dataset:")
print(confusion_matrix_NN2)

NN_pred2=NN_Classifier3.predict(USPSNumeralsInputMat)
y_pred = accuracy_score(USPSNumeralsTarMat,NN_pred)
majority_voting_usps.append(NN_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_NN3 = confusion_matrix(USPSNumeralsTarMat,NN_pred2)
print('Implementation 6: Testing Accuracy for Neural Network Using USPS Dataset:'+str(y_pred))
print("Implementation 6: Confusion Matrix for Neural Network Using USPS Dataset:")
print(confusion_matrix_NN3)


# ## Implementation 7: SVM using USPS

# In[142]:


X_test, y_test = USPSNumeralsInputMat, USPSNumeralsTarMat
SVM_pred1=SVM_classifier1.predict(X_test)
y_pred = accuracy_score(y_test,SVM_pred1)
majority_voting_usps.append(SVM_pred1)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_SVM1 = confusion_matrix(y_test,SVM_pred1)
print('Implementation 7: Testing Accuracy for SVM Using USPS Dataset:'+str(y_pred))
print("Implementation 7: Confusion Matrix for SVM Using USPS Dataset:")
print(confusion_matrix_SVM1)


# ## Implementation 8 & 9: SVM using USPS

# In[ ]:


X_test, y_test = USPSNumeralsInputMat, USPSNumeralsTarMat
SVM_pred2=SVM_classifier2.predict(X_test)
y_pred = accuracy_score(y_test,SVM_pred2)
majority_voting_usps.append(SVM_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_SVM2 = confusion_matrix(y_test,SVM_pred2)
print('Implementation 8: Testing Accuracy for SVM Using USPS Dataset:'+str(y_pred))
print("Implementation 8: Confusion Matrix for SVM Using USPS Dataset:")
print(confusion_matrix_SVM2)


# In[ ]:


X_test, y_test = USPSNumeralsInputMat, USPSNumeralsTarMat
SVM_pred3=SVM_classifier3.predict(X_test)
y_pred = accuracy_score(y_test,SVM_pred3)
majority_voting_usps.append(SVM_pred3)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_SVM3 = confusion_matrix(y_test,SVM_pred3)
print('Implementation 9: Testing Accuracy for SVM Using USPS Dataset:'+str(y_pred))
print("Implementation 9: Confusion Matrix for SVM Using USPS Dataset:")
print(confusion_matrix_SVM3)


# In[ ]:


X_test, y_test = USPSNumeralsInputMat, USPSNumeralsTarMat
SVM_pred4=SVM_classifier4.predict(X_test)
y_pred = accuracy_score(y_test,SVM_pred4)
majority_voting_usps.append(SVM_pred4)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_SVM4 = confusion_matrix(y_test,SVM_pred4)
print('Implementation 10: Testing Accuracy for SVM Using USPS Dataset:'+str(y_pred))
print("Implementation 10: Confusion Matrix for SVM Using USPS Dataset:")
print(confusion_matrix_SVM4)


# ## Implementation 10 & 11 & 12: Random Forest using USPS

# In[139]:


X_test, y_test = USPSNumeralsInputMat, USPSNumeralsTarMat

RF_pred1=RF_classifier1.predict(X_test)
y_pred = accuracy_score(y_test,RF_pred1)
majority_voting_usps.append(RF_pred1)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_RF1 = confusion_matrix(y_test,RF_pred1)
print('Implementation 10: Testing Accuracy for Random Forest Using USPS Dataset:'+str(y_pred))
print("Implementation 10: Confusion Matrix for Random Forest Using USPS Dataset:")
print(confusion_matrix_RF1)

RF_pred2=RF_classifier2.predict(X_test)
y_pred = accuracy_score(y_test,RF_pred2)
majority_voting_usps.append(RF_pred2)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_RF2 = confusion_matrix(y_test,RF_pred2)
print('Implementation 11: Testing Accuracy for Random Forest Using USPS Dataset:'+str(y_pred))
print("Implementation 11: Confusion Matrix for Random Forest Using USPS Dataset:")
print(confusion_matrix_RF2)

RF_pred3=RF_classifier3.predict(X_test)
y_pred = accuracy_score(y_test,RF_pred3)
majority_voting_usps.append(RF_pred3)
print("Majority List: " +  str(np.matrix(majority_voting_usps).shape))
confusion_matrix_RF3 = confusion_matrix(y_test,RF_pred3)
print('Implementation 12: Testing Accuracy for Random Forest Using USPS Dataset:'+str(y_pred))
print("Implementation 12: Confusion Matrix for Random Forest Using USPS Dataset:")
print(confusion_matrix_RF3)


# ## Implementation 13: CNN on USPS data set

# In[141]:


(x_test, y_test) = (np.array(USPSNumeralsInputMat),USPSNumeralsTarMat)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = x_test.reshape(USPSNumeralsInputMat.shape[0], img_x, img_y, 1)
score = CNN_model.evaluate(x_test, y_test, verbose=0)
CNN_model_pred_usps = CNN_model.predict(np.array(x_test))
CNN_predicted_value_usps = CNN_model_pred_usps.argmax(axis=-1)
confusion_matrix_CNN2 = confusion_matrix(CNN_predicted_value_usps,USPSNumeralsTarMat)
loss,accuracy = CNN_model.evaluate(x_test, y_test, verbose=False)
majority_voting_usps.append(CNN_predicted_value_usps)
print("Majority List: " + str(np.matrix(majority_voting_usps).shape))
print('Implementation 13: Testing Accuracy for CNN Using USPS Dataset:', score[1])
print("Implementation 13: Confusion Matrix for CNN Using USPS Dataset:")
print(confusion_matrix_CNN2)


# ## Majority Voting on USPS data set

# In[143]:


final_majority_usps_list = np.matrix(majority_voting_usps)
final_majority_usps_list = mode(final_majority_usps_list)
print("Final Majority USPS List: " +str((final_majority_usps_list[0][0]).shape))
y_pred = accuracy_score(y_test,final_majority_usps_list[0][0])
MV1_confusion_matrix = confusion_matrix(y_test,final_majority_usps_list[0][0])
print('Testing Accuracy for Majority Voting Using USPS Dataset:', str(y_pred))
print("Confusion Matrix for Majority Voting  Using USPS Dataset:")
print(MV1_confusion_matrix)


# ## Bagging for USPS Dataset Using Library

# In[178]:


x_test, y_test = (np.array(USPSNumeralsInputMat), USPSNumeralsTarMat)

scores = cross_val_score(bagging1, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,bagging1.predict(x_test))
print(MV_confusion_matrix) 

scores = cross_val_score(bagging2, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,bagging2.predict(x_test))
print(MV_confusion_matrix) 

scores = cross_val_score(bagging3, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,bagging3.predict(x_test))
print(MV_confusion_matrix) 

scores = cross_val_score(bagging4, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,bagging4.predict(x_test))
print(MV_confusion_matrix) 


# ## Boosting for USPS Dataset Using Library

# In[177]:


x_test, y_test = (np.array(USPSNumeralsInputMat), USPSNumeralsTarMat)

scores = cross_val_score(boosting1, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,boosting1.predict(x_test))
print(MV_confusion_matrix) 


scores = cross_val_score(boosting2, x_test, y_test, cv=5, scoring='accuracy')
print(scores.mean())
MV_confusion_matrix = confusion_matrix(y_test,boosting2.predict(x_test))
print(MV_confusion_matrix) 

