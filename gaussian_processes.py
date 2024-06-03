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

    # length scale = 1 and noise level = 1 initially but remove them for hyperparameter optimization
    # RBF(length_scale=3.83) + WhiteKernel(noise_level=1.16e-05)
    ker = RBF() + WhiteKernel()
    reg = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10, copy_X_train=False)
    reg.fit(X_train[:3000], Y_train[:3000])

    print(reg.kernel_)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    #residual calculations and plotting
    Y_train_pred, Y_train_pred_std = reg.predict(X_train[-1000:],return_std=True) #a)
    plt.figure(figsize=(12,5)) #a)
    plt.plot(Y_train[-1000:]) #a)
    plt.title('prediction on the training set')
    Y_train_pred, Y_train_pred_std = reg.predict(X_train[-1000:],return_std=True) #a)
    plt.errorbar(np.arange(len(X_train[-1000:])), (Y_train_pred), yerr=2*Y_train_pred_std,fmt='.r') #a)
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred'])#a)
    # save plot
    plt.savefig('./plots/training_set_prediction.png')
    plt.show() #a)

    plt.figure(figsize=(12,5)) #a)
    plt.title('prediction on the validation set')
    plt.plot(Y_val[-1000:]) #a)
    Y_val_pred, Y_val_pred_std = reg.predict(X_val[-1000:],return_std=True) #a)
    plt.errorbar(np.arange(len(X_val[-1000:])), (Y_val_pred), yerr=2*Y_val_pred_std,fmt='.r') #a)
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) #a)

    # RMSE computation
    RMSE = np.sqrt(np.mean((Y_val_pred - Y_val[-1000:])**2))

    # save plot
    plt.savefig(f'./plots/validation_set_prediction_RMSE{RMSE}.png')
    plt.show() #a)

    print(f'Validation RMSE= {RMSE}')

    model_now = reg
    fmodel = lambda u,y: model_now.predict(np.concatenate([u,y])[None,:])[0]
    Y_test_sim = use_NARX_model_in_simulation(u[-(int(len(u)*test_size)):], fmodel, na, nb)

    plt.plot(Y_test[-500:]) #b)
    plt.plot(Y_test_sim[-500:]) #b)
    plt.grid(); plt.xlabel('index time'); plt.ylabel('y'); plt.legend(['measured','prediction']) #b)
    
    # RMSE computation
    RMSE = np.sqrt(np.mean((Y_test_sim[-500:] - Y_test[-500:])**2))
    # save plot
    plt.savefig(f'./plots/test_set_prediction_NRMS{RMSE}.png')
    plt.show() #b)

    print('Test RMSE =', RMSE) #b)


if __name__ == '__main__':
    run()