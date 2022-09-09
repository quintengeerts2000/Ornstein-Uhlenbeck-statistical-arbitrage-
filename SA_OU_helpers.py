from tqdm import trange, tqdm
import pandas as pd
from numpy.linalg import eig, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.optimize as so
import numpy as np

def compute_log_likelihood(params, *args):
    '''
    Compute the average Log Likelihood, this function will by minimized by scipy.
    Find in (2.2) in linked paper

    returns: the average log likelihood from given parameters
    '''
    # functions passed into scipy's minimize() needs accept one parameter, a tuple of
    #   of values that we adjust to minimize the value we return.
    #   optionally, *args can be passed, which are values we don't change, but still want
    #   to use in our function (e.g. the measured heights in our sample or the value Pi)
    m_i, kappa, sigma = params
    X, dt = args
    n = len(X)

    sigma_tilde_squared = sigma ** 2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    summation_term = 0

    for i in range(1, len(X)):
        summation_term += (X[i] - X[i - 1] * np.exp(-kappa * dt) - m_i * (1 - np.exp(-kappa * dt))) ** 2

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    log_likelihood = (-np.log(2 * np.pi) / 2) + (-np.log(np.sqrt(sigma_tilde_squared))) + summation_term

    return -log_likelihood
    # since we want to maximize this total log likelihood, we need to minimize the
    #   negation of the this value (scipy doesn't support maximize)

def estimate_coefficients_MLE(X, dt, tol=1e-10):
    '''
    Estimates Ornstein-Uhlenbeck coefficients (θ, µ, σ) of the given array
    using the Maximum Likelihood Estimation method

    input: X - array-like time series data to be fit as an OU process
           dt - time increment (1 / days(start date - end date))
           tol - tolerance for determination (smaller tolerance means higher precision)
    returns: θ, µ, σ, Average Log Likelihood
    '''

    bounds = ((None, None), (1e-5, None), (1e-5, None))  # m_i ∈ ℝ, kappa > 0, sigma > 0
                                                           # we need 1e-10 b/c scipy bounds are inclusive of 0, 
                                                           # and sigma = 0 causes division by 0 error
    m_init = np.mean(X)
    initial_guess = (m_init, 10, 1)  # initial guesses for m_i, kappa, sigma
    result = so.minimize(compute_log_likelihood, initial_guess, args=(X, dt), bounds=bounds, tol=tol)
    m, kappa, sigma = result.x 
    sigma_eq = sigma / np.sqrt(2*kappa)
    return kappa, m, sigma, sigma_eq

def estimate_coefficients_LR(data, LOOKBACK_PERIOD):
    '''
    Estimates Ornstein-Uhlenbeck coefficients (θ, µ, σ) of the given array
    using the Linear regression method'''
    #fitting the model
    X = data[:-1].reshape(-1,1)
    y = data[1:]
    model2 = LinearRegression().fit(X,y)
    a = model2.intercept_
    b = model2.coef_
    pred = model2.predict(X)
    zeta = y - pred
    # OU parameters 
    kappa = -np.log(b) * LOOKBACK_PERIOD
    m = a/(1-b)
    sigma = np.sqrt(np.var(data)*2*kappa)
    sigma_eq = sigma / np.sqrt(2*kappa)
    return kappa[0], m[0], sigma[0], sigma_eq[0]

def standardise_returns(returns:pd.DataFrame, LOOKBACK_PERIOD:int, epochs:int) -> pd.DataFrame:
    standardised = returns.copy() #create a copy just to keep the structure of the dataset
    for epoch in tqdm(range(0,epochs)):
        start_range = epoch*LOOKBACK_PERIOD
        end_range = (epoch+1)*LOOKBACK_PERIOD
        standardised.iloc[start_range:end_range,:] = ((returns.iloc[start_range:end_range,:] - returns.iloc[start_range:end_range,:].mean())/ returns.iloc[start_range:end_range,:].std())
    standardised = standardised.iloc[0:end_range,:] #cut the final incomplete epoch out of the data
    return standardised

def construct_eigenportfolios(standardised:pd.DataFrame,returns:pd.DataFrame, LOOKBACK_PERIOD:int, epochs:int, EIGENPORTFOLIO_CUTOFF:float) -> pd.DataFrame:
    #constructing the eigenportfolio's:

    #initialising the dataframe
    eig_portf = pd.DataFrame(index=range(1,epochs),columns=['portfolios','num_of_portfolios']) # for each individual epoch the eigenportfolio will be saved

    #the first eigenportfolio can be constructed only after the first epoch is complete
    for epoch in tqdm(range(1,epochs)):
        #selecting the appropriate data
        start_range = (epoch-1)*LOOKBACK_PERIOD
        end_range = (epoch)*LOOKBACK_PERIOD
        std_ret_of_epoch = standardised.iloc[start_range:end_range,:]

        #calculating correlation matrix and calculating eigenvalues/eigenvectors
        corr_of_epoch = std_ret_of_epoch.corr()
        corr_matrix_of_epoch = corr_of_epoch.to_numpy()
        values_of_epoch, vectors_of_epoch = eig(corr_matrix_of_epoch)

        #selecting the amount of eigenportfolios based on the threshhold
        num_of_vectors = 1
        tot_var_of_portf = values_of_epoch[0]
        tot_var = values_of_epoch.sum()
        while tot_var_of_portf / tot_var < EIGENPORTFOLIO_CUTOFF:
            num_of_vectors += 1
            tot_var_of_portf += values_of_epoch[num_of_vectors-1]
        vectors_of_epoch = vectors_of_epoch
        
        #normalisation of portfolio weights
        eigportf_of_epoch = vectors_of_epoch / returns.iloc[start_range:end_range,:].std().values[:,None]
        eigportf_of_epoch = eigportf_of_epoch / norm(eigportf_of_epoch,1)
        
        #save the portfolios
        eig_portf['portfolios'].loc[epoch] = eigportf_of_epoch
        eig_portf['num_of_portfolios'].loc[epoch] = num_of_vectors
    return eig_portf

def calculate_ret_eigportf(returns:pd.DataFrame, eig_portf:pd.DataFrame, epochs:int, LOOKBACK_PERIOD:int):
    #calculation of the returns of all the eigenportfolios keep in mind for the calculation of the stationary time series only the first x-amount are needed
    ret_eigportf_forward = pd.DataFrame(columns= range(len(returns.columns)), index=returns.iloc[0:LOOKBACK_PERIOD].index) #return of the portfolio at time [t,t+L] trained on [t-L,t] data 
    ret_eigportf_training = pd.DataFrame(columns= range(len(returns.columns))) #return of the portfolio at time [t-L,t] trained on [t-L,t] data, same as used in exploration

    for epoch in tqdm(range(1,epochs-1)):
        #selecting the appropriate data
        start_range_training = (epoch-1)*LOOKBACK_PERIOD #time t-L
        end_range_training = (epoch)*LOOKBACK_PERIOD #time t

        start_range_forward = (epoch)*LOOKBACK_PERIOD #time t
        end_range_forward = (epoch+1)*LOOKBACK_PERIOD #time t+L

        portf = eig_portf.loc[epoch,'portfolios']
        ret_eigportf_forward = ret_eigportf_forward.append(returns.iloc[start_range_forward:end_range_forward,:].dot(portf))
        ret_eigportf_training = ret_eigportf_training.append(returns.iloc[start_range_training:end_range_training,:].dot(portf)) 

    ret_eigportf_training = ret_eigportf_training.append(pd.DataFrame(columns= range(len(returns.columns)), index=returns.iloc[(epochs-1)*LOOKBACK_PERIOD:epochs*LOOKBACK_PERIOD].index)) #add this so the index of the dataframes stay consistent
    return ret_eigportf_training, ret_eigportf_forward

def create_stationary_timeseries(returns:pd.DataFrame, standardised:pd.DataFrame, epochs:int,LOOKBACK_PERIOD:int, ret_eigportf_training:pd.DataFrame,ret_eigportf_forward:pd.DataFrame, eig_portf:pd.DataFrame):
    goodness_of_fit = pd.DataFrame(columns=['R2_of_train', 'R2_of_forward']) #keep this information

    R2_training_all_epochs = pd.DataFrame(columns=returns.columns,index=range(1,epochs-1))
    R2_forward_all_epochs = pd.DataFrame(columns=returns.columns,index=range(1,epochs-1))
    betas_all_epochs = pd.DataFrame(columns=returns.columns,index=range(1,epochs-1))
    alphas_all_epochs = pd.DataFrame(columns=returns.columns,index=range(1,epochs-1))

    returns_res_forward = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns) #stationary time series we want to construct
    returns_res_training = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns)

    LR = LinearRegression(fit_intercept=False) #using a linear regression model to find the betas
    LR2 = LinearRegression(fit_intercept=True)
    tot_count = -1 #needed during runtime to append to a dataframe

    for epoch in tqdm(range(1,epochs-1)):
        #selecting the appropriate data
        start_range_training = (epoch-1)*LOOKBACK_PERIOD #time t-L
        end_range_training = (epoch)*LOOKBACK_PERIOD #time t

        start_range_forward = (epoch)*LOOKBACK_PERIOD #time t
        end_range_forward = (epoch+1)*LOOKBACK_PERIOD #time t+L

        #make sure you use the eigenportfolios of the current epoch
        eigen_portf_of_epoch = int(eig_portf.num_of_portfolios.loc[epoch]) #eigenportfolio for the period [t,t+L], trained on data of [t-L,t]
        
        #constructing the model for each epoch for each asset
        for asset in returns.columns:

            #training the standardised accounting for drift data of [t-L,t]
            y_training = standardised[asset].iloc[start_range_training:end_range_training].values.reshape(LOOKBACK_PERIOD, 1)
            x_training = ((ret_eigportf_training.iloc[start_range_training:end_range_training] - ret_eigportf_training.iloc[start_range_training:end_range_training].mean())/ret_eigportf_training.iloc[start_range_training:end_range_training].std()).values[:,0:eigen_portf_of_epoch]
            LR.fit(x_training,y_training) #fit the data

            #training the asset model on data of [t-L,t], outputs used are the betas and the alpha necessary for the modified s_score
            y_training_asset = returns[asset].iloc[start_range_training:end_range_training].values.reshape(LOOKBACK_PERIOD, 1)
            x_training_asset = ret_eigportf_training.iloc[start_range_training:end_range_training].values[:,0:eigen_portf_of_epoch]
            LR2.fit(x_training_asset, y_training_asset)

            pred_training = LR.predict(x_training) #check how well the model fits on the training data
            R2_training = r2_score(y_training, pred_training)

            #forward data used as a check to see how well the model fits on future data, what i hope is that the model stays fairly consistent
            y_forward = standardised[asset].iloc[start_range_forward:end_range_forward].values.reshape((LOOKBACK_PERIOD, 1))
            x_forward = ((ret_eigportf_forward.iloc[start_range_forward:end_range_forward] - ret_eigportf_forward.iloc[start_range_forward:end_range_forward].mean())/ret_eigportf_forward.iloc[start_range_forward:end_range_forward].std()).values[:,0:eigen_portf_of_epoch]
            
            pred_forward = LR.predict(x_forward)
            R2_forward = r2_score(y_forward, pred_forward)

            #return of the residuals are to be saved for time [t,t+L], undo the standardisation used for fitting the portfolios
            ret_forw = (returns[asset].iloc[start_range_forward:end_range_forward].values - (pred_forward*returns[asset].iloc[start_range_forward:end_range_forward].std() + returns[asset].iloc[start_range_forward:end_range_forward].mean()).flatten())
            ret_train = (returns[asset].iloc[start_range_training:end_range_training].values - (pred_training*returns[asset].iloc[start_range_training:end_range_training].std() + returns[asset].iloc[start_range_training:end_range_training].mean()).flatten())

            #save these returns in the appropriate dataframes
            returns_res_forward.iloc[start_range_forward:end_range_forward, returns_res_forward.columns.get_loc(asset)] = ret_forw
            returns_res_training.iloc[start_range_training:end_range_training, returns_res_training.columns.get_loc(asset)] = ret_train

            #save the goodness of fit
            tot_count += 1
            goodness_of_fit.loc[tot_count] = [R2_training, R2_forward]
            R2_training_all_epochs.loc[epoch,asset] = R2_training
            R2_forward_all_epochs.loc[epoch,asset] = R2_forward
            betas_all_epochs.loc[epoch,asset] = LR2.coef_
            alphas_all_epochs.loc[epoch,asset] = LR2.intercept_[0]

    return goodness_of_fit, R2_training_all_epochs, R2_forward_all_epochs, returns_res_forward, returns_res_training, betas_all_epochs,alphas_all_epochs

def calculate_xi(ret_eigportf_training, returns,returns_res_forward,returns_res_training,LOOKBACK_PERIOD,epochs):
    Xi_forward = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns)
    Xi_training = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns)
    #excess will be calculated every lookback period
    for epoch in tqdm(range(1,epochs-1)):
        #selecting the appropriate data
        start_range_training = (epoch-1)*LOOKBACK_PERIOD #time t-L
        end_range_training = (epoch)*LOOKBACK_PERIOD #time t

        start_range_forward = (epoch)*LOOKBACK_PERIOD #time t
        end_range_forward = (epoch+1)*LOOKBACK_PERIOD #time t+L

        #calculating the excess for every epoch period
        Xi_forward.iloc[start_range_forward:end_range_forward,:] = (returns_res_forward.iloc[start_range_forward:end_range_forward,:]+1).cumprod()
        Xi_training.iloc[start_range_training:end_range_training,:] = (returns_res_training.iloc[start_range_training:end_range_training,:]+1).cumprod()
    return Xi_training, Xi_forward

def calculate_OU_params(returns,Xi_training, epochs, LOOKBACK_PERIOD):
    OU_params = pd.DataFrame(index=range(1,epochs-1), columns=returns.columns)
    #OU parameters will be calculated every epoch
    for epoch in tqdm(range(1,epochs-1)):
        #selecting the appropriate data
        start_range_training = (epoch-1)*LOOKBACK_PERIOD #time t-L
        end_range_training = (epoch)*LOOKBACK_PERIOD #time t

        #calculating the OU-parameters on the training data for every epoch
        for asset in returns.columns:
            kappa_epoch, m_epoch, sigma_epoch, sigma_eq_epoch = estimate_coefficients_LR(Xi_training.iloc[start_range_training:end_range_training,Xi_training.columns.get_loc(asset)].values, LOOKBACK_PERIOD)
            #save the parameters
            OU_params.loc[epoch,asset] = (kappa_epoch, m_epoch, sigma_epoch, sigma_eq_epoch)
    
    return OU_params

def calculate_s_scores(returns, ret_eigportf_training, OU_params,alphas_all_epochs, Xi_forward, Xi_training, LOOKBACK_PERIOD, epochs, DRIFT_SENSITIVITY):
    s_score_forward = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns)
    s_score_training = pd.DataFrame(index=ret_eigportf_training.index, columns=returns.columns)

    #OU parameters will be calculated every epoch
    for epoch in tqdm(range(1,epochs-1)):
        #selecting the appropriate data
        start_range_training = (epoch-1)*LOOKBACK_PERIOD #time t-L
        end_range_training = (epoch)*LOOKBACK_PERIOD #time t

        start_range_forward = (epoch)*LOOKBACK_PERIOD #time t
        end_range_forward = (epoch+1)*LOOKBACK_PERIOD #time t+L

        #calculating the OU-parameters on the training data for every epoch
        for asset in returns.columns:
            m_epoch = OU_params.loc[epoch, asset][1]
            sigma_eq_epoch = OU_params.loc[epoch, asset][3]
            kappa_epoch = OU_params.loc[epoch, asset][0]
            alpha_epoch = alphas_all_epochs.loc[epoch,asset]

            Xi_forward_epoch = Xi_forward.iloc[start_range_forward:end_range_forward,Xi_training.columns.get_loc(asset)].values
            Xi_training_epoch = Xi_training.iloc[start_range_training:end_range_training,Xi_training.columns.get_loc(asset)].values

            #calculate the s_score per epoch
            s_score_forward.iloc[start_range_forward:end_range_forward,Xi_training.columns.get_loc(asset)] =  (Xi_forward_epoch - m_epoch) / sigma_eq_epoch - (alpha_epoch * LOOKBACK_PERIOD / (kappa_epoch*sigma_eq_epoch))* DRIFT_SENSITIVITY
            s_score_training.iloc[start_range_training:end_range_training,Xi_training.columns.get_loc(asset)] =  (Xi_training_epoch - m_epoch) / sigma_eq_epoch - (alpha_epoch * LOOKBACK_PERIOD / (kappa_epoch*sigma_eq_epoch))*DRIFT_SENSITIVITY
    
    return s_score_training, s_score_forward

def prepare_data(returns_path, log_returns_path, LOOKBACK_PERIOD, EIGENPORTFOLIO_CUTOFF,DRIFT_SENSITIVITY):
    returns = pd.read_csv(returns_path).set_index('close_time') #load data
    log_returns = pd.read_csv(log_returns_path).set_index('close_time')
    standardised = returns.copy() #create a copy just to keep the structure of the dataset

    epochs = int(np.floor(len(returns) / LOOKBACK_PERIOD)) #define the mount of epochs, each epoch has length LOOKBACK_PERIOD

    #create a dataframe with the standardised returns for each epoch
    standardised = standardise_returns(returns,LOOKBACK_PERIOD,epochs)
    #create the eigenportfolios for each epoch
    eig_portf = construct_eigenportfolios(standardised,returns, LOOKBACK_PERIOD, epochs, EIGENPORTFOLIO_CUTOFF)
    ret_eigportf_training, ret_eigportf_forward = calculate_ret_eigportf(returns,eig_portf, epochs, LOOKBACK_PERIOD)
    goodness_of_fit, R2_training_all_epochs, R2_forward_all_epochs, returns_res_forward, returns_res_training, betas_all_epochs, alphas_all_epochs = create_stationary_timeseries(returns, standardised, epochs,LOOKBACK_PERIOD, ret_eigportf_training,ret_eigportf_forward, eig_portf)
    Xi_training, Xi_forward = calculate_xi(ret_eigportf_training, returns,returns_res_forward,returns_res_training,LOOKBACK_PERIOD,epochs)
    OU_params = calculate_OU_params(returns,Xi_training, epochs, LOOKBACK_PERIOD)
    s_score_training, s_score_forward = calculate_s_scores(returns, ret_eigportf_training, OU_params,alphas_all_epochs, Xi_forward, Xi_training, LOOKBACK_PERIOD, epochs, DRIFT_SENSITIVITY)
    
    return returns,log_returns, goodness_of_fit, R2_training_all_epochs, R2_forward_all_epochs, returns_res_forward, returns_res_training, Xi_training, Xi_forward, OU_params, eig_portf, s_score_training, s_score_forward, betas_all_epochs,alphas_all_epochs,ret_eigportf_training, ret_eigportf_forward