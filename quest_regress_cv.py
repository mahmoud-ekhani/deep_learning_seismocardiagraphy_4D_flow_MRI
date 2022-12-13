import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as CB
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Rescaling 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomZoom
import numpy as np
import os
import tensorflow.keras as keras
import random
import scipy.io
import h5py
seed = 100
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
def create_mlp(dim,regress=False):
    # define the Multilayer Perceptron (MLP) network:
    model = Sequential()
    model.add(Dense(8,input_dim=dim,activation="relu"))
    model.add(Dense(4,activation="relu"))
    # check if a regression node needs to be added 
    if regress:
        model.add(Dense(1,activation="linear"))
    # return the model
    return model
def create_cnn(image_size=(256,256,3),filters=(64,128,256,512),augmentation=False,regress=False):
    # initialize the input shape and channel dimension
    inputShape = image_size
    chanDim = -1
    # define the model input
    inputs = Input(shape=(inputShape))
    # data augmentation
    if augmentation:
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.2)(x)
    # loop over the filters
    for i, filter in enumerate(filters):
        # if this is the first layer, set the input appropriately
        if i == 0 and not augmentation:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(filter,kernel_size=3,padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=2)(x)
    # flatten the volume, then FC=>RELU=>BN=>DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply anothe layer of FC layer, this one matches the number of nodes
    # coming out of the MLP 
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check if the regression node is active
    if regress:
        x = Dense(1,activation="linear")(x)
    # construct the CNN
    model = Model(inputs=inputs, outputs=x)
    # return the CNN model
    return model
def load_mat_file(mat_file):
    f = h5py.File(mat_file+'.mat','r')
    variables = f.items()
    for var in variables:
        name = var[0]
        data = var[1]
        if type(data) is not h5py.Dataset:
            continue
        if str(name)=="attr":
            attr = data
            attr = np.array(attr,dtype=np.float32)
            attr = np.transpose(attr,(1,0))
        elif str(name)=="vmax":
            vmax = data
            vmax = np.array(vmax,dtype=np.float32)
            vmax = np.transpose(vmax,(1,0))
        elif str(name)=="scalogram":
            scg = data
            scg = np.array(scg,dtype=np.float32)
            scg = np.transpose(scg,(2,1,0))
            sh  = np.shape(scg)
            scg = np.resize(scg,(sh[0],sh[1],sh[2],1))
    return scg, attr, vmax
callback = CB.ModelCheckpoint("mixed_mdl_reg.keras",
                             save_best_only=True, 
                              monitor="val_loss")
for iterations in range(1,6):
    # read the training set
    scg_train,atr_train,reg_train=load_mat_file(os.path.join('scg_dataset','train_cv_'+str(iterations)))
    # return the predictors of the training set
    train_x = [atr_train,scg_train]
    # return the labels of training set normalized with the vmax in training set
    train_y = reg_train/np.max(reg_train)
    # read the test set
    scg_test,atr_test,reg_test=load_mat_file(os.path.join('scg_dataset','test_cv_'+str(iterations)))
    # return the predictors of test set
    test_x = [atr_test,scg_test]
    # return the labels of test set normalized with the vmax in training set
    test_y = reg_test/np.max(reg_train)
    # create the multi-layer perceptron model based on the patient's attributes
    mlp = create_mlp(atr_train.shape[1],regress=False)
    # create the cnn model based on the scalogram images
    cnn = create_cnn(image_size=(256,256,1),filters=(4,8,16,32,64,128,256),
                     augmentation=False,regress=False)
    # combine the mlp and cnn models
    combInp = keras.layers.concatenate([mlp.output,cnn.output])
    x = keras.layers.Dense(4,activation="relu")(combInp)
    x = keras.layers.Dense(1,activation="linear")(x)
    model = keras.Model(inputs=[mlp.input,cnn.input],outputs=x)
    # compile the mixture model using mean absolute percentage error as loss function
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="mean_absolute_percentage_error",optimizer=opt)
    # train the model
    history = model.fit(x=train_x, 
                  y=train_y,
                  validation_data = (test_x, test_y),
                  epochs=100,
                  batch_size=32, 
                  verbose=2,
                  callbacks=callback)
    # load the model at the epoch with the lowest error over the validation set
    model = tf.keras.models.load_model("mixed_mdl_reg.keras")
    # predict the vmax in test set
    preds = model.predict(test_x,verbose=2)
    # return the unique vmax values within the test set
    unique_vmax = np.unique(test_y)
    # instantiate a numpy array to record the ground-truth vs. predicted vmax for each subj
    vmax_gt_pd = list()
    for v in unique_vmax:
        gt_vmax = test_y[test_y==v]*np.max(reg_train)
        pd_vmax = preds[test_y==v]*np.max(reg_train)
        vmax_gt_pd.append([gt_vmax,pd_vmax])
    # save vmax_gt_pd as .mat file
    filename1 = 'cv_'+str(iterations)+'.mat'
    scipy.io.savemat(filename1,{'vmax':vmax_gt_pd})
