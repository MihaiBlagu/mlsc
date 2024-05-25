from data_utils.data_utils import load_data, create_IO_data, split_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import os
from matplotlib import pyplot as plt 


def use_NARX_model_in_simulation(ulist, f, na, nb):
    #init upast and ypast as lists.
    upast = [0]*nb 
    ypast = [0]*na 
    
    ylist = []
    for unow in ulist:
        #compute the current y given by f
        ynow = f(upast,ypast) 
        
        #update past arrays
        upast.append(unow)
        upast.pop(0)
        ypast.append(ynow)
        ypast.pop(0)
        
        #save result
        ylist.append(ynow)
    return np.array(ylist) #return result


def run():
    # Load data
    file_path = 'disc-benchmark-files/training-val-test-data.npz'
    u, th = load_data(file_path)

    # Create input-output pairs
    na = 5
    nb = 5
    X, Y = create_IO_data(u, th, na, nb)

    # Split data into training, validation, and testing sets
    val_size = 0.2
    test_size = 0.1
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, val_size, test_size)

    # print('X_train:', X_train.shape)
    # print('X_val:', X_val.shape)
    # print('X_test:', X_test.shape)
    # print('Y_train:', Y_train.shape)
    # print('Y_val:', Y_val.shape)
    # print('Y_test:', Y_test.shape)

    ker = RBF(length_scale=1) + WhiteKernel(noise_level=1)
    reg = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10, copy_X_train=False)
    reg.fit(X_train[:1000], Y_train[:1000])

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    #residual calculations and plotting
    Y_train_pred, Y_train_pred_std = reg.predict(X_train,return_std=True) #a)
    plt.figure(figsize=(12,5)) #a)
    plt.plot(Y_train) #a)
    plt.title('prediction on the training set')
    Y_train_pred, Y_train_pred_std = reg.predict(X_train,return_std=True) #a)
    plt.errorbar(np.arange(len(X_train)), (Y_train_pred), yerr=2*Y_train_pred_std,fmt='.r') #a)
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred'])#a)
    # save plot
    plt.savefig('./plots/training_set_prediction.png')
    plt.show() #a)

    plt.figure(figsize=(12,5)) #a)
    plt.title('prediction on the validation set')
    plt.plot(Y_val) #a)
    Yval_pred, Yval_pred_std = reg.predict(X_val,return_std=True) #a)
    plt.errorbar(np.arange(len(X_val)), (Yval_pred), yerr=2*Yval_pred_std,fmt='.r') #a)
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) #a)

    NRMS = np.mean((Yval_pred-Y_val)**2)**0.5/np.std(Y_val)

    # save plot
    plt.savefig(f'./plots/validation_set_prediction_NRMS{NRMS}.png')
    plt.show() #a)

    print(f'Validation NRMS= {NRMS}')#a)

    # np.random.seed(43)
    # utest = np.random.normal(scale=1.0,size=5000)
    # ytest = use_NARX_model_in_simulation(utest,f,na,nb)


    model_now = reg
    fmodel = lambda u,y: model_now.predict(np.concatenate([u,y])[None,:])[0] 
    Y_test_sim = use_NARX_model_in_simulation(X_test, fmodel, na, nb)

    plt.plot(Y_test) #b)
    plt.plot(Y_test-Y_test_sim) #b)
    plt.grid(); plt.xlabel('index time'); plt.ylabel('y'); plt.legend(['measured','prediction']) #b)
    
    NRMS = np.mean((Y_test-Y_test_sim)**2)**0.5/np.std(Y_test)
    # save plot
    plt.savefig(f'./plots/test_set_prediction_NRMS{NRMS}.png')
    plt.show() #b)

    print('NRMS=', NRMS) #b)


if __name__ == '__main__':
    run()