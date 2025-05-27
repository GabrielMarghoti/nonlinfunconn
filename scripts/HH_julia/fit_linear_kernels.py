import numpy as np
from scipy.optimize import root, minimize, least_squares
from scipy.signal import convolve
import matplotlib.pyplot as plt

def eci(x,p,power_t=None):
        '''Evaluate 
        '''
        # 
        if not isinstance(p,mf.struct.irrarray):
            if power_t is None:
                ecj = pp.ExponentialConvolution(p[1:],p[0])
            else:
                # power_t tells you the power of the power of t in terms that
                # are like t^n exp(-gt)
                ecj = pp.ExponentialConvolution(p[1],p[0])
                if power_t[1] > 0:
                    for q in np.arange(power_t[1]): ecj.convolve_exp(p[1])
                for h in np.arange(len(p)-2):
                    for q in np.arange(power_t[h]+1): ecj.convolve_exp(p[h])
        else:
            if power_t is None:
                ecj = pp.ExponentialConvolution(p(branch=0)[1:],p(branch=0)[0])
                for b in np.arange(len(p.first_index["branch"])-1)[1:]:
                    branch_par = p(branch=b)
                    ecj.branch_path(branch_par[1],branch_par[0])
                    for h in np.arange(len(branch_par))[2:]:
                        ecj.convolve_exp(branch_par[h],branch=b)
            else:
                p_b0 = p(branch=0)
                pt_b0 = power_t(branch=0) #power_t must also be an irrarray
                
                ecj = pp.ExponentialConvolution([p_b0[1]],p_b0[0])
                if pt_b0[1] > 0:
                    for q in np.arange(pt_b0[1]): ecj.convolve_exp(p_b0[1])
                for h in np.arange(len(p_b0))[2:]:
                    for q in np.arange(pt_b0[h]+1): ecj.convolve_exp(p[h])
                
                for b in np.arange(len(p.first_index["branch"])-1)[1:]:
                    branch_par = p(branch=b)
                    pt_bi = power_t(branch=b)
                    ecj.branch_path(branch_par[1],branch_par[0])
                    if pt_bi[1]>0: 
                        for q in np.arange(pt_bi[1]): ecj.convolve_exp(branch_par[1],branch=b)
                    for h in np.arange(len(branch_par))[2:]:
                        for q in np.arange(pt_bi[h]+1):
                            ecj.convolve_exp(branch_par[h],branch=b)
                
        y = ecj.eval(x)
        del ecj
        
        return y
    
    def fit_eci(x,y,n_min=3,n_max=5,method=None,routine="least_squares",g1=None,g2=None,rms_limits=[None,None],auto_stop=False,rms_tol=1e-2):
        
        params = []
        rms = []
        
        y_norm = np.nansum(y)
        if np.isnan(y_norm) or np.isinf(y_norm) or y_norm==0:
            return None, None
        else:
            yb = y/y_norm
        
        for i in np.arange(n_min,n_max+1):
            p0 = 0.2+np.arange(i)*0.03
            p0 = np.append(1.,p0)
            
            lower_bounds = [-np.inf]
            for q in np.arange(i): lower_bounds.append(0.0)
            upper_bounds = [np.inf for q in np.arange(i+1)]
            if g1 is not None:
                p0[1] = g1
                lower_bounds[1] = g1*0.9
                upper_bounds[1] = g1*1.1
            if g2 is not None:
                p0[2] = g2
                lower_bounds[2] = g2*0.9
                upper_bounds[2] = g2*1.1
            
            if routine == "minimize":
                error = lambda p,x,y: np.sum(np.power(linear_exponential_kernel(x,p)-y,2))
                res = minimize(error,p0,args=(x,yb),method=method)
                p = res.x
            elif routine == "least_squares":
                residuals = lambda p,x,y: linear_exponential_kernel(x,p) - y
                res = least_squares(residuals,p0,args=(x,yb),method=method,bounds=(lower_bounds,upper_bounds))
                p = res.x
            p[0] *= y_norm
            params.append(p)
            rms.append(np.sqrt(np.sum(np.power((linear_exponential_kernel(x,p)-y)[rms_limits[0]:rms_limits[1]],2))))
            
            if auto_stop and i>n_min:
                delta_rms_rel = np.abs(rms[-1]-rms[-2])/rms[-2]
                if delta_rms_rel<rms_tol: break
        
        return params,rms



# Define the Heaviside step function (theta function)
def heaviside(t):
    return np.where(t >= 0, 1.0, 0.0)

# Define the exponential kernel component
def exp_component(t, gamma):
    return heaviside(t) * np.exp(-gamma * t)

# Define the general kernel function for two components
def linear_exponential_kernel(t, c1, gamma1_0, gamma1_1, c2, gamma2_0, gamma2_1):
    # First kernel component
    kernel1 = exp_component(t, gamma1_0)
    kernel1 = convolve(kernel1, exp_component(t, gamma1_1), mode='same')
    
    # Second kernel component
    kernel2 = exp_component(t, gamma2_0)
    kernel2 = convolve(kernel2, exp_component(t, gamma2_1), mode='same')
    
    # Total kernel is the sum of both components
    return c1 * kernel1 + c2 * kernel2

# Convolution function to fit
def model(t, c1, gamma1_0, gamma1_1, c2, gamma2_0, gamma2_1, x):
    # Calculate kernel for the time points
    k = linear_exponential_kernel(t, c1, gamma1_0, gamma1_1, c2, gamma2_0, gamma2_1)
    
    # Convolve the input signal x with the kernel
    return convolve(x, k, mode='same')

# Sample neuron time-series data for fitting (t, y_obs, and x)
t = np.linspace(0, 10, 100)
# Simulated input signal (x)
x = np.exp(-0.5 * t) * np.sin(t)  # Example input signal
# Simulated output signal (y) based on a known kernel with noise
true_params = [1.0, 0.5, 0.2, 0.5, 0.3, 0.1]  # True parameters for the kernel
y_obs = model(t, [1.0, 0.5, 0.2, 0.5, 0.3, 0.1], x) + 0.05 * np.random.normal(size=t.size)  # Add noise

# Initial guess for parameters: c1, gamma1_0, gamma1_1, c2, gamma2_0, gamma2_1
initial_guess = [1.2, 0.6, 0.1, 0.1, 0.1, 0.2]  # Adjust based on expected behavior

# Fit the kernel
def fit_kernel(t, y_obs, x, initial_guess):
    # Fit the model function to data using non-linear least squares
    popt, pcov = curve_fit(lambda t, *params: model(t, *params, x), t, y_obs, p0=initial_guess)
    return popt, pcov

# Perform fitting
popt, pcov = fit_eci(x, y_obs, initial_guess)

# Generate fitted data using the optimized parameters
fitted_y = model(t, *popt, x)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, y_obs, label='Observed Data (y)', marker='o', linestyle='-', markersize=4, color='blue')
plt.plot(t, fitted_y, label='Fitted Kernel Response (y)', linestyle='-', color='orange')
plt.title('Fitting Linear Exponential Kernel to Neuron Response Data')
plt.xlabel('Time')
plt.ylabel('Neuron Response')
plt.legend()
plt.grid()
plt.show()

# Display fitted parameters
print("Fitted parameters:", popt)
