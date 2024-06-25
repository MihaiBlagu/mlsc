from data_utils.data_utils import load_data, create_IO_data, split_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
import numpy as np
import os
from matplotlib import pyplot as plt 
import warnings
from scipy.interpolate import interp1d
import joblib
import itertools
warnings.filterwarnings("ignore")


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


def get_model(x,y): 
    #return a regressor which is fitted to x,y
    ker = RBF() + WhiteKernel() + ExpSineSquared() #a)
    gp_reg = GaussianProcessRegressor(ker, n_restarts_optimizer=10) #a)
    gp_reg.fit(x[:,None],y) #a)
    return gp_reg #a)

def get_mean_std(gp_reg, xtest_points):
    ytest_pred_mean, ytest_pred_std = gp_reg.predict(xtest_points[:,None],return_std=True) #a)
    ytest_std_mean = (ytest_pred_std**2 - np.exp(gp_reg.kernel_.k2.theta))**0.5
    return ytest_pred_mean, ytest_std_mean
    
def acquisition_var(gp_reg, xtest_points):
    ytest_pred_mean, ytest_pred_std = get_mean_std(gp_reg, xtest_points) #b)
    return ytest_pred_std**2 #b)

def acquisition_weighted_mean_and_std(gp_reg, xtest_points, weight=0.5):
    ytest_pred_mean, ytest_pred_std_mean = get_mean_std(gp_reg, xtest_points) #d)    
    return (1-weight)*ytest_pred_mean + weight*ytest_pred_std_mean #d)

def bayesian_optimization(u, th, acquisition_fun, n_initial=5, n_max=15, seed=22):
    # f : is the function which need to be sampled
    # xmin : and xmax are the bounds on the x
    # acquisition_fun(gp_reg, some_x_points) : is the acquisition_fun on which the maximum need to be chosen as next point
    # n_initial : the number of points which are uniformly sampled from f before using bayesian optimizaiton
    # n_max : the buget of the number of maximum points that can be sampled from f 
    # (i.e. n_initial - n_max is the number of bayesian samples)
    x = u[:n_initial] #c=)
    y = th[:n_initial] #c=)
    xtest_points = u[1000:] #c=)
    for n in range(n_initial+1, n_max+1): #c)
        gp_reg = get_model(x,y)
        acquisition_vals = acquisition_fun(gp_reg, xtest_points) #c=)
        xnew = xtest_points[np.argmax(acquisition_vals)] #c=)
        ynew = th[np.where(u == xnew)][0] #c=)
        x = np.append(x,xnew) #c=)
        y = np.append(y,ynew) #c=)
    return x, y, get_model(x,y)

def run_bayesian_opt():
    # Load data
    file_path = 'disc-benchmark-files/training-val-test-data.npz'
    u, th = load_data(file_path)

    n_initial = 5
    n_max = 15

    x_rand = u[1000:1015] #random baseline
    y_rand = th[1000:1015]
    x_test = u[-6000:-1000]

    weight = 0.8
    #incorporate the weight factor in the function with a lambda function
    acquisition_weighted_mean_and_std_now = lambda gp_reg, xtest_points: \
        acquisition_weighted_mean_and_std(gp_reg, xtest_points, weight=weight)

        
    for mode,acquisition_fun in enumerate([acquisition_var,acquisition_weighted_mean_and_std_now]):
        if acquisition_fun==None:
            continue
        if mode==0:
            print('Variance Acquision')
        else:
            print(f'Weighted mean and Variance Acquision (weight={weight})')
        #Bayesian
        x, y, reg = bayesian_optimization(u, th, acquisition_fun=acquisition_fun, n_initial=n_initial, n_max=n_max, seed=21)

        plt.figure(figsize=(12,4))
        for i,(xi, yi) in enumerate([(x_rand,y_rand),(x,y)]):
            plt.subplot(1,2,i+1)
            plt.plot(x_test,th[-6000:-1000],label='real')

            label = 'random samples' if i==0 else 'bayesian optimization'
            plt.plot(xi,yi,'o',label=label)

            reg = get_model(xi,yi)

            ytest_pred_mean, ytest_pred_std_mean = get_mean_std(reg, x_test)
            plt.plot(x_test, ytest_pred_mean,label='mean')
            plt.fill_between(x_test, \
                            ytest_pred_mean+1.92*ytest_pred_std_mean,\
                            ytest_pred_mean-1.92*ytest_pred_std_mean,\
                            alpha=0.2,label='92% std function')
            plt.grid()
            plt.legend()
            plt.ylabel('y')
            plt.xlabel('x')
        plt.show()
        
        if mode==1:
            M = x_test[np.argmax(th[-6000:-1000])]
            x_near = np.linspace(M-0.05, M+0.05,num=300)
            ytest_pred_mean, ytest_pred_std_mean = get_mean_std(reg, x_near)

            # added
            x_th = np.linspace(M-0.05, M+0.05, num=len(th[-6000:-1000]))
            f = interp1d(x_th, th[-6000:-1000], kind='linear')
            th_resampled = f(x_near)    
            
            plt.plot(x_near, th_resampled, label='real')
            xlim, ylim = plt.xlim(), plt.ylim()
            plt.plot(xi,yi,'o',label='bayesian samples')
            plt.plot(x_near, ytest_pred_mean,label='mean')
            plt.fill_between(x_near, \
                            ytest_pred_mean+1.92*ytest_pred_std_mean,\
                            ytest_pred_mean-1.92*ytest_pred_std_mean,\
                            alpha=0.2,label='92% std function')
            plt.xlim(xlim); plt.ylim(ylim[0],ylim[1]+0.005)
            plt.legend()
            plt.grid()
            plt.ylabel('y')
            plt.xlabel('x')
            plt.show()


def frequency_domain_analysis(file_path):
    # Load data
    u, th = load_data(file_path)
    
    # Perform FFT on input and output signals
    U = np.fft.fft(u)
    TH = np.fft.fft(th)
    
    # Compute the frequency bins
    N = len(u)
    T = 1.0 / N  # assuming a normalized sample rate of 1 for simplicity
    freqs = np.fft.fftfreq(N, T)
    
    # Plot the magnitude spectrum
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(U), label='|U(f)|')
    plt.title('Input Voltage (u) - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.abs(TH), label='|TH(f)|')
    plt.title('Output Theta (th) - Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Optionally, plot phase spectrum
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.angle(U), label='Phase U(f)')
    plt.title('Input Voltage (u) - Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.angle(TH), label='Phase TH(f)')
    plt.title('Output Theta (th) - Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def load_and_plot(model_path='./models/gaussian_process_model.pkl', na=5, nb=5):
    # Load data
    file_path = 'disc-benchmark-files/training-val-test-data.npz'
    u, th = load_data(file_path)

    # Create input-output pairs
    X, Y = create_IO_data(u, th, na, nb)

    # Split data into training, validation, and testing sets
    val_size = 0.2
    test_size = 0.1
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, val_size, test_size)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    reg = joblib.load(model_path)

    


def train_gp_instance(na, nb, ker):
    # Load data
    file_path = 'disc-benchmark-files/training-val-test-data.npz'
    u, th = load_data(file_path)

    X, Y = create_IO_data(u, th, na, nb)
    val_size = 0.2
    test_size = 0.1
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X[:5000], Y[:5000], val_size, test_size)

    # print(f'Training with na={na}, nb={nb}')
    # print('X_train:', X_train.shape)
    # print('X_val:', X_val.shape)
    # print('X_test:', X_test.shape)
    # print('Y_train:', Y_train.shape)
    # print('Y_val:', Y_val.shape)
    # print('Y_test:', Y_test.shape)

    reg = GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=10, copy_X_train=False)
    reg.fit(X_train, Y_train)

    Y_val_pred, Y_val_pred_std = reg.predict(X_val, return_std=True)
    val_rmse = np.sqrt(np.mean((Y_val_pred - Y_val)**2))
    print(f'Kernel: {reg.kernel_}, Validation RMSE: {val_rmse}')

    Y_test_pred, Y_test_pred_std = reg.predict(X_test, return_std=True)
    test_rmse = np.sqrt(np.mean((Y_test_pred - Y_test)**2))
    print(f'Test RMSE: {test_rmse}')

    return reg, val_rmse, test_rmse

def grid_search_gp():
    na_values = [3, 5, 7]
    nb_values = [3, 5, 7]
    kernels = [
        RBF() + WhiteKernel(),
        ExpSineSquared() + WhiteKernel(),
        RBF() + ExpSineSquared() + WhiteKernel()
    ]

    best_rmse_val = [float('inf'), None]
    best_rmse_test = [float('inf'), None]

    results_file = './models/kernels_results.txt'

    for na, nb, ker in itertools.product(na_values, nb_values, kernels):
        reg, val_rmse, test_rmse = train_gp_instance(na, nb, ker)

        with open(results_file, 'a') as f:
            f.write(f'na={na}, nb={nb}, Kernel: {reg.kernel_}, Validation RMSE: {val_rmse}, Test RMSE: {test_rmse}\n')
            print(f'na={na}, nb={nb}, Kernel: {reg.kernel_}, Validation RMSE: {val_rmse}, Test RMSE: {test_rmse}')

        if val_rmse < best_rmse_val[0]:
            best_rmse_val[0] = val_rmse
            best_rmse_val[1] = (reg, na, nb, ker)
            print(f"New val best: {best_rmse_val[0]} with na={na}, nb={nb}, kernel={ker}")

        if test_rmse < best_rmse_test[0]:
            best_rmse_test[0] = test_rmse
            best_rmse_test[1] = (reg, na, nb, ker)
            print(f"New test best: {best_rmse_test[0]} with na={na}, nb={nb}, kernel={ker}")

    # Save the best models and their details to the results file
    best_reg_val, best_na_val, best_nb_val, best_kernel_val = best_rmse_val[1]
    joblib.dump(best_reg_val, f'./models/best_model_val_rmse_{best_rmse_val[0]}.pkl')

    best_reg_test, best_na_test, best_nb_test, best_kernel_test = best_rmse_test[1]
    joblib.dump(best_reg_test, f'./models/best_model_test_rmse_{best_rmse_test[0]}.pkl')

    with open(results_file, 'a') as f:
        f.write(f'\nBest Validation RMSE: {best_rmse_val[0]} with na={best_na_val}, nb={best_nb_val}, kernel={best_kernel_val}\n')
        f.write(f'Best Test RMSE: {best_rmse_test[0]} with na={best_na_test}, nb={best_nb_test}, kernel={best_kernel_test}\n')

    print(f'Best Validation RMSE: {best_rmse_val[0]} with na={best_na_val}, nb={best_nb_val}, kernel={best_kernel_val}')
    print(f'Best Test RMSE: {best_rmse_test[0]} with na={best_na_test}, nb={best_nb_test}, kernel={best_kernel_test}')



def test_gp(model_path='./models/best_model_test_rmse_0.002982304576111068.pkl', na=7, nb=3):
    file_path = 'disc-benchmark-files/training-val-test-data.npz'
    u, th = load_data(file_path)
    
    # Create input-output pairs
    X, Y = create_IO_data(u, th, na, nb)

    val_size = 0.2
    test_size = 0.1
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X[:5000], Y[:5000], val_size, test_size)

    model = joblib.load(model_path)

    # #residual calculations and plotting
    # Y_train_pred, Y_train_pred_std = model.predict(X_train[-1000:],return_std=True) #a)
    # plt.figure(figsize=(12,5)) #a)
    # plt.plot(Y_train[-1000:]) #a)
    # plt.title('prediction on the training set')
    # Y_train_pred, Y_train_pred_std = model.predict(X_train[-1000:],return_std=True) #a)
    # plt.errorbar(np.arange(len(X_train[-1000:])), (Y_train_pred), yerr=2*Y_train_pred_std,fmt='.r') #a)
    # plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred'])#a)
    # # save plot
    # plt.savefig('./plots/training_set_prediction.png')
    # plt.show() #a)

    plt.figure(figsize=(12,5)) #a)
    plt.title('Prediction on the Test set')
    plt.plot(Y_test[-1000:]) #a)
    Y_test_pred, Y_test_pred_std = model.predict(X_test[-1000:],return_std=True) #a)
    plt.errorbar(np.arange(len(X_test[-1000:])), (Y_test_pred), yerr=2*Y_test_pred_std,fmt='.r') #a)
    plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) #a)

    # RMSE computation
    RMSE = np.sqrt(np.mean((Y_test_pred - Y_test[-1000:])**2))
    print(f'Validation Prediction RMSE= {RMSE}')
    # save plot
    plt.savefig(f'./plots/final_val_pred{RMSE}.png')
    plt.show() #a)
    
    # plt.figure(figsize=(12,5)) #a)
    # plt.title('Prediction on test set')
    # plt.plot(X_test) #a)
    # Y_test_pred, Y_test_pred_std = model.predict(X_test,return_std=True) #a)
    # plt.errorbar(np.arange(len(X_test)), (Y_test_pred), yerr=2*Y_test_pred_std,fmt='.r') #a)
    # plt.grid(); plt.xlabel('sample'); plt.ylabel('y'); plt.legend(['measured','pred']) #a)

    # # RMSE computation
    # RMSE = np.sqrt(np.mean((Y_test_pred - Y_test)**2))
    # print(f'Test Prediciton RMSE= {RMSE}')
    # # save plot
    # plt.savefig(f'./plots/final_test_pred{RMSE}.png')
    # plt.show() #a)

    fmodel = lambda u,y: model.predict(np.concatenate([u,y])[None,:])[0]
    # Y_test_sim = use_NARX_model_in_simulation(u[-(int(len(u)*test_size)):], fmodel, na, nb)
    u = u[:5000]
    Y_test_sim = use_NARX_model_in_simulation(u[-X_test.shape[0]:], fmodel, na, nb)

    # Shift Y_test_sim 10 positions to the left and pad with zeros
    shift_amount = 7
    cutoff_index = len(Y_test_sim) - shift_amount
    Y_test_sim_shifted = np.zeros_like(Y_test_sim)
    if len(Y_test_sim) > shift_amount:
        Y_test_sim_shifted[:cutoff_index] = Y_test_sim[shift_amount:]

    plt.figure(figsize=(12,5))
    plt.title('Simulation on the Test set')
    plt.plot(Y_test[:cutoff_index]) #b)
    plt.plot(Y_test_sim_shifted[:cutoff_index]) #b)
    plt.grid(); plt.xlabel('index time'); plt.ylabel('y'); plt.legend(['measured','prediction']) #b)
    
    # RMSE computation
    RMSE = np.sqrt(np.mean((Y_test_sim_shifted[:cutoff_index] - Y_test[:cutoff_index])**2))
    # save plot
    plt.savefig(f'./plots/final_test_sim{RMSE}.png')
    plt.show() #b)

    print('Test RMSE =', RMSE) #b)


if __name__ == '__main__':
    # grid_search_gp()
    test_gp()
    # run_bayesian_opt()
    # frequency_domain_analysis('disc-benchmark-files/training-val-test-data.npz')