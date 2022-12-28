import numpy as np
import matplotlib.pyplot as plt

def exp_decay(x, A, tau):
    return A*np.exp(-x/tau)  

def with_append():
    tau = 2
    times = np.linspace(0,10,num=100)
    A_vals = np.linspace(0.5,4,num=8)
    
    W = np.empty((times.shape[0],0)) #make an empty array with times.shape[0] rows and 0 columns
    for i in range(0,A_vals.size,1):
        A_val = A_vals[i]
        y_vals = exp_decay(times, A_val, tau)
        y_vals = y_vals[:,np.newaxis] #https://stackoverflow.com/a/33481152
        W = np.append(W, y_vals, axis=1) #add the new column
    
    return W, times, A_vals, tau

def with_meshgrid():
    tau = 2
    times = np.linspace(0,10,num=100)
    A_vals = np.linspace(0.5,4,num=8)
    
    t_mg, A_mg = np.meshgrid(times, A_vals)
    W = exp_decay(t_mg, A_mg, tau)
    W = W.T #transpose so that W works with the plot_code function
    
    return W, times, A_vals, tau
    
def plot_code(W, times, A_vals, tau):
    plt.figure()
    for i in range(0,W.shape[1],1):
        y_vals = W[:,i]
        A_val = A_vals[i]
        plt.plot(times, y_vals, label='$A$ = '+str(A_val))
    plt.xlabel('$t$')
    plt.ylabel(r'$A$ exp($-t/\tau$)')
    plt.title(r'Exponential decay with $\tau$' + ' = {:.3f}'.format(tau))
    plt.legend()
    plt.show() 

if __name__ == "__main__":
    W, times, A_vals, tau = with_append()
    plot_code(W, times, A_vals, tau)
    
    W, times, A_vals, tau = with_meshgrid()
    plot_code(W, times, A_vals, tau)
    