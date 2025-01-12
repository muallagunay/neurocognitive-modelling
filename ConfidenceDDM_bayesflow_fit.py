#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:03:27 2024

@author: mualla
"""

import numpy as np
import os
import pandas as pd
# from tqdm.autonotebook import tqdm 
import bayesflow as bf
from numba import njit
from scipy.stats import truncnorm, norm, ttest_1samp, t
import seaborn as sns
import matplotlib.pyplot as plt
from modules.pyhddmjagsutils import m_plot_recovery, recovery, recovery_scatter, plot_posterior2d


model_name = 'ConfidenceDDM_v2_to_pe'
train_fitter = False
fit_data = True

# Function to save data
def save_data(df, subject_means, name, file_path=None):
    """
    Save the DataFrame with the parameter estimations to a CSV file.

    Args:
    - df (pd.DataFrame): The full DataFrame.
    - subject_means (dict): The subject's data to append.
    - file_path (str): The path to the CSV file where the DataFrame is saved.

    Returns:
    - pd.DataFrame: The updated DataFrame.
    """
    new_row = pd.DataFrame(subject_means, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    if file_path != None:
        full_file_path = os.path.join(file_path, f"{name}.csv")
        df.to_csv(full_file_path, index=False)    
    return df

def confidence_interval(means, stds, n, confidence=0.95):
    """
    Calculate confidence intervals for each column using means and standard deviations.

    Parameters:
    - means: Series containing means for each column
    - stds: Series containing standard deviations for each column
    - n: Sample size for each column
    - confidence: Confidence level (default is 0.95 for 95% confidence interval)

    Returns:
    - intervals: DataFrame containing the confidence intervals for each column
    """
    std_errs = stds / np.sqrt(n)  # Standard error of the mean
    t_value = t.ppf((1 + confidence) / 2, n - 1)  # T-score for the given confidence level

    margin_of_error = t_value * std_errs

    lower_bound = means - margin_of_error
    upper_bound = means + margin_of_error
    
    return lower_bound, upper_bound

def credible_interval (post_mean, post_std):
    ci = norm.ppf([0.025, 0.975], loc=post_mean, scale=post_std)
    lower_boundary = ci[0]
    upper_boundary = ci[1]
    return lower_boundary, upper_boundary
    
# Function to generate a prior distribution for all model parameters
def prior_N(n_min=60, n_max=300):
    """A prior for the random number of observation"""
    return np.random.randint(n_min, n_max+1)

# For easier definition of truncated normals
def truncnorm_better(mean=0, sd=1, low=-10, upp=10, size=1):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)

def draw_prior():
    
    """ 
    Setting prior distributions.
    
    Drift(0), Boundary(1), Beta (2; start point), Tau (3; non-decision time), Eta (4; variance |e| trials), 
    Confidence drift(5), Confidence Boundary(6), Confidence Tau(7), Confidence Eta (8), 
    Sigma (9; Variance in eeg), Confidence Drift Slope(10), Confidence Drift Intercept(11)  

    """
        
    # mu_drift ~ N(0, 2.0) # mean of drift rate
    mu_drift = np.random.normal(0.0, 2.0) 

    # boundary  ~ N(1.0, 0.25) in [0, 2] # boundary
    boundary = truncnorm_better(mean=1.0, sd=0.25, low=0.0, upp=2)[0]
    
    # beta ~ Beta(2.0, 2.0)  # relative start point
    beta = np.random.beta(2.0, 2.0)
    
    # tau ~ N(0.5, 0.125) in [0, 1] # non-decision time
    tau = truncnorm_better(mean=0.5, sd=0.125, low=0.0, upp=1)[0]
        
    # conf_mu_drift ~ N(0, 2.0) # mean post-decisional drift rate (v2)
    conf_mu_drift =  np.random.normal(0.0, 2.0) 
        
    # conf_boundary ~  N(1.0, 0.25) in [0, 2] # confidence boundaries
    conf_boundary  = truncnorm_better(mean=1.0, sd=0.25, low=0.0, upp=2)[0]
    
    # conf_tau ~ N(0.5, 0.125) in [0, 1] # confidence non-decision time
    conf_tau = truncnorm_better(mean=0.3, sd=0.125, low=-1, upp=1)[0]
        
    # sigma ~ N(0, 0.5) in [0, 10] # variance in Pe (eeg)
    sigma = truncnorm_better(mean=0, sd=0.5, low=0.0, upp=10)[0] 

    # v2_slope ~ N(0, 5) in [-1, 10]  # v2 slope
    v2_slope = truncnorm_better(mean=0, sd=5, low=-1.0, upp=10)[0] 
    
    # v2_intercept ~ N(0, 3.0) # v2 intercept
    v2_intercept = np.random.normal(0, 3.0)
    
    p_samples = np.hstack((mu_drift, boundary, beta, tau, conf_mu_drift, conf_boundary, conf_tau, sigma, v2_slope, v2_intercept))
    return p_samples

num_params = len(draw_prior())

# Define the model in simulation 
@njit
def diffusion_trial(mu_drift, boundary, beta, tau,
                    conf_mu_drift, conf_boundary, conf_tau, sigma, v2_slope, v2_intercept,
                    dc=1.0, dt=.005):
    
    """Simulates a trial from the diffusion model."""

    evidence = boundary * beta


######### Decision #########
    n_steps = 0.
    
    # trial-to-trial drift rate variability
    drift_trial = mu_drift
    
    # Simulate a single choice DM path
    while (evidence > 0 and evidence < boundary):

        # DDM equation
        evidence += drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()

        # Increment step
        n_steps += 1.0

    choicert = n_steps * dt + tau

    if evidence >= boundary:
        choice =  1  # choice A
    elif evidence <= 0:
        choice = -1  # choice B
    else:
        choice = 0  # This indicates a missing response

######### Confidence #########
    n_conf_steps = 0.
    
    # trial-to-trial drift rate variability
    conf_drift_trial = conf_mu_drift
    
    #Pe
    pe = np.random.normal(v2_intercept + v2_slope * conf_drift_trial, sigma)
    
    if choice == 1:
        while ((evidence < boundary + conf_boundary/2) and (evidence > boundary - conf_boundary/2)):
    
            # DDM equation
            evidence += conf_drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()
    
            # Increment step
            n_conf_steps += 1.0
    
        conf_rt = n_conf_steps * dt + conf_tau
    
        if evidence >= boundary + conf_boundary/2:
            conf =  1  # high confidence
        elif evidence <= boundary - conf_boundary/2:
            conf = -1  # low confidence
        else:
            conf = 0  # This indicates a missing response
    
    elif choice  == -1 :
        while ((evidence < conf_boundary/2) and (evidence > -conf_boundary/2)):
    
            # DDM equation
            evidence += conf_drift_trial*dt + np.sqrt(dt) * dc * np.random.normal()
    
            # Increment step
            n_conf_steps += 1.0
    
        conf_rt = n_conf_steps * dt + conf_tau
    
        if evidence <= - conf_boundary/2:
            conf =  1  # high confidence
            
        elif evidence >= conf_boundary/2:
            conf = -1  # low confidence
            
        else:
            conf = 0  # This indicates a missing response
       
    return choicert, choice, conf_rt, conf, pe

@njit
def simulate_trials(params, n_trials):
    """Simulates a diffusion process for trials."""

    mu_drift, boundary, beta, tau, conf_mu_drift, conf_boundary, conf_tau, sigma, v2_slope,  v2_intercept  = params
    choicert = np.empty(n_trials)
    conf_rt = np.empty(n_trials)
    choice = np.empty(n_trials)
    conf = np.empty(n_trials)
    pe = np.empty(n_trials)
    
    for i in range(n_trials):
        choicert[i], choice[i], conf_rt[i], conf[i], pe[i] = diffusion_trial(mu_drift, boundary, beta, tau, conf_mu_drift, conf_boundary, conf_tau, sigma, v2_slope, v2_intercept)
    
    sim_data = np.stack((choicert, choice, conf_rt, conf, pe), axis=-1)
    return sim_data

# Connecting priors and simulations
# Takes the priors
prior = bf.simulation.Prior(prior_fun=draw_prior) 
experimental_context = bf.simulation.ContextGenerator(non_batchable_context_fun=prior_N) 

# Simulates the performance of a subject in a whole experiment given a set of parameter values and context variables
simulator = bf.simulation.Simulator(simulator_fun=simulate_trials, 
    context_generator=experimental_context)

# Connects the priors and the simulator
generative_model = bf.simulation.GenerativeModel(prior, simulator) 


# Create Configurator
# Configurator is used for a better set-up of network training in bayesflow. 
# We need this, since the variable N cannot be processed directly by the nets so we want to use log N.
def configurator(sim_dict):
    """Configures the outputs of a generative model for interaction with 
    BayesFlow modules."""
    
    out = dict()
    # These will be passed through the summary network. In this case,
    # it's just the data, but it can be other stuff as well.
    data = sim_dict['sim_data'].astype(np.float32)
    out['summary_conditions'] = data
    
    # These will be concatenated to the outputs of the summary network
    # Convert N to log N since neural nets cant deal well with large numbers
    N = np.log(sim_dict['sim_non_batchable_context'])
    # Repeat N for each sim (since shared across batch), notice the
    # extra dimension needed
    N_vec = N * np.ones((data.shape[0], 1), dtype=np.float32)
    out['direct_conditions'] = N_vec
    
    # Finally, extract parameters. Any transformations (e.g., standardization)
    # should happen here.
    if sim_dict['prior_draws'] is not None:
        out['parameters'] = sim_dict['prior_draws'].astype(np.float32)
    return out

# BayesFlow Setup
summary_net = bf.networks.InvariantNetwork() # learns the most informative summary stats from data 
inference_net = bf.networks.InvertibleNetwork(num_params=num_params) # from summary statistics to posterior distributions - joint posterior distribution 
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net) # connects the two networks

# Create the checkpoint path 
checkpoint_path = f"checkpoint/{model_name}"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Connect the networks (summary and inference) with the generative model 
trainer = bf.trainers.Trainer(
    amortizer=amortizer, # networks
    generative_model=generative_model,  #prior/posterior predictives. learns the full possibilities of what the predictive data would be
    configurator=configurator,# transform data + prior 
    checkpoint_path=checkpoint_path) # save the trained net

if train_fitter:
    """Create validation simulations with some random N, if specific N is desired, need to 
    call simulator explicitly or define it with keyword arguments which can control behavior
    All trainer.train_*** can take additional keyword arguments controling the behavior of
    configurators, generative models and networks"""
    num_val = 300
    val_sims = generative_model(num_val)

    # Experience-replay training
    losses = trainer.train_experience_replay(epochs= 1500,
                                                 batch_size=32,
                                                 iterations_per_epoch=1000,
                                                 validation_sims=val_sims)
    
else:
    status = trainer.load_pretrained_network()
    print("Loaded the pre-trained network.")


plot_path = f"recovery_plots/{model_name}"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Fit empirical data 

# Load data
file_path = './data/BY/combined_data.csv'
empirical_data = pd.read_csv(file_path)

# Clean data
#Rename
empirical_data.columns = ["cj", "cor", "rt", "choice", "rt2", "eeg", "Subject"]
empirical_data = empirical_data.drop('choice', axis=1)
#Re-order
empirical_data = empirical_data[["Subject", "rt", "cor", "rt2", "cj",  "eeg"]]
# Confidence = -1 for 1,2,3 | Confidence = 1 for 4,5,6
empirical_data["cj"] = empirical_data["cj"].replace({1: -1, 2: -1, 3: -1, 4: 1, 5: 1, 6: 1})
# Choice = 0 is -1 | choice = 1 is 1
empirical_data["cor"].replace({0: 1, 1: -1}, inplace=True)

# Standardize the Pe data
eeg_mean = empirical_data['eeg'].mean()
eeg_std = empirical_data['eeg'].std()
empirical_data['Stand_eeg'] = (empirical_data['eeg'] - eeg_mean) / eeg_std

save_to_path = './results/BY/v2_to_pe_model'
if not os.path.exists(save_to_path):
    os.makedirs(save_to_path)

if fit_data:
    
    # No prior given
    input_params = None

    # Estimated parameters (their posterior distributions)
    param_names = ['Drift Rate', 'Boundary', 'Beta', 'NDT', 'Confidence Drift Rate',
                    'Confidence Boundary', 'Confidence NDT', 'Sigma', 'Slope', 'Intercept']

    # Initialize a dataframes to save the posterior means, standard deviations, and variance explained
    posterior_means = pd.DataFrame()
    posterior_stds = pd.DataFrame()
    SDR_total = pd.DataFrame()
    credible_intervals = pd.DataFrame()

    # Fit data for each subject seperately
    for subj in range (1,17): # subjects 1 to 16
            
        # Sub-set data
        obs_data = empirical_data[empirical_data['Subject'] == subj] 
        obs_data = obs_data.drop(columns=["Subject", "eeg"]) #remove the subject number and the original eeg (so it fits standardized eeg)
        obs_data = obs_data.values        
        n_trials = obs_data.shape[0]
        obs_dict = {'sim_data': obs_data[np.newaxis,:,:], 
        'sim_non_batchable_context': n_trials, 'prior_draws': input_params} 
        configured_dict = configurator(obs_dict) 
        
        # Obtain posterior samples
        num_posterior_draws = 10000
        post_samples = amortizer.sample(configured_dict, num_posterior_draws) # posterior samples from the real data
        print(f'The posterior means are {np.mean(post_samples,axis=0)}')
                    
        # Mean of posteriors
        post_means = np.mean(post_samples,axis=0)
        subject_means = dict(zip(param_names, post_means))                
        posterior_means = save_data(posterior_means, subject_means, 'mean_param_posteriors', save_to_path)
        
        # Standard deviation of posteriors
        post_std = np.std(post_samples, axis=0)       
        subject_stds = dict(zip(param_names, post_std))                
        posterior_stds = save_data(posterior_stds, subject_stds, 'standard_deviation_param_posteriors',  save_to_path)
        
        lower_boundary, upper_boundary = credible_interval (post_means[8], post_std[8]) 
        subj_credible_interval = {'95% Credible Interval Lower': lower_boundary,
                               '95% Credible Interval Upper': upper_boundary
                               }
        
        credible_intervals = save_data(credible_intervals, subj_credible_interval, 'credible_interval_v2_slope', save_to_path)
                    
        # Plot each subject's v2_slope posterior
        v2_slope_posterior = post_samples[:, 8]
        save_plot = os.path.join(save_to_path, "v2_posteriors/")
        if not os.path.exists(save_plot):
            os.makedirs(save_plot)

        # Plot the values
        plt.figure()
        plt.hist(v2_slope_posterior, bins=20, edgecolor='black')
        plt.xlabel('Estimated Value')
        plt.ylabel('Draws')
        plt.title(f"V2 Slope Posterior Draws Subject {subj}")
        plt.savefig(save_plot+ f"sub-{subj}.png")
        plt.close()
                
# Plotting predicted and observed data
obs_data = empirical_data
obs_data = obs_data.drop(columns=["Subject", "eeg"]) #remove the subject number and the original eeg (so it fits standardized eeg)
obs_data = obs_data.values        
n_trials = obs_data.shape[0]
obs_dict = {'sim_data': obs_data[np.newaxis,:,:], 
'sim_non_batchable_context': n_trials, 'prior_draws': input_params} 
configured_dict = configurator(obs_dict)

predicted_data_list = []

for i in range (100):
    post_samples = amortizer.sample(configured_dict, 1) 
    post_samples = np.hstack(post_samples)
    predict_data = simulate_trials(post_samples, 50).astype(np.float32)
    predicted_data_list.append(predict_data)

predicted_data_array = np.array(predicted_data_list)
predicted_data_2d = predicted_data_array.reshape(-1, predicted_data_array.shape[2])

obs_data_df = pd.DataFrame(obs_data)
pred_data_df = pd.DataFrame(predicted_data_2d)

obs_data_df.to_csv(os.path.join(save_to_path, 'obs_data_bayesflow.csv'), index = False)
pred_data_df.to_csv(os.path.join(save_to_path,'pred_data_bayesflow.csv'), index = False)

# if plot_estim_data: # I use this if I want to plot on Python. Now I am saving df because I plot on R. 
    
#     save_plots = os.path.join(save_to_path, "sim_vs_observed.png")
#     fig, axes = plt.subplots(1, 3, figsize=(15,7), tight_layout=True)
    
#     sns.kdeplot(1000*predicted_data_2d[:,0].flatten(), ax = axes[0], linewidth = 3)
#     axes[0].hist(1000*obs_data[:,0], density=True, bins = 80, alpha=0.8)
#     axes[0].set_title('Choice RT', fontsize=14)
#     axes[0].set_xlabel('Decision Reaction Time (msec)', fontsize=14)
#     axes[0].set_ylabel('Density', fontsize=14)
#     axes[0].set_xlim([0,2000])   
    
#     sns.kdeplot(1000*predicted_data_2d[:,2].flatten(), ax = axes[1], linewidth = 3)
#     axes[1].hist(1000*obs_data[:,2],density=True, bins = 80, alpha=0.8)
#     axes[1].set_title('Confidence RT', fontsize=14)
#     axes[1].set_xlabel('Confidence Reaction Time (msec)', fontsize=14)
#     axes[1].set_ylabel('Density', fontsize=14)
#     axes[1].set_xlim([-200,2000])
    
#     sns.kdeplot(1000*predicted_data_2d[:,4].flatten(), ax = axes[2], linewidth = 3)
#     axes[2].hist(1000*obs_data[:,4], density=True, bins = 80, alpha=0.8)
#     axes[2].set_title('Pe Amplitude', fontsize=14)
#     axes[2].set_xlabel('Pe Amplitude', fontsize=14)
#     axes[2].set_ylabel('Density', fontsize=14)
#     axes[2].set_xlim([-2000,2000])
    
#     fig.savefig(save_plots)

### Report results ###

# Means of all parameters
parameter_means = posterior_means.mean()
parameter_stds = posterior_means.std()
parameter_lower_ci , parameters_upper_ci = confidence_interval(parameter_means, parameter_stds, n = 31, confidence=0.95)

parameter_stats  = { 'Parameter': parameter_means.index,
                    'Mean':parameter_means,
                 'STD': parameter_stds,
                 'Lower CI': parameter_lower_ci,
                 'Upper CI': parameters_upper_ci }

parameter_stats_df = pd.DataFrame(parameter_stats)
save_path = os.path.join(save_to_path, "parameter_info.csv")
parameter_stats_df.to_csv(save_path, index = False)

# V2 Per participant
v2_means = posterior_means['Slope']
v2_stds = posterior_stds['Slope']
v2_slope_post_info = pd.concat([v2_means, v2_stds, credible_intervals], axis = 1)
v2_slope_post_info.insert(0, 'Mean', v2_slope_post_info.iloc[:, 0]) 
v2_slope_post_info.insert(1, 'STD', v2_slope_post_info.iloc[:, 2]) 
v2_slope_post_info = v2_slope_post_info.drop(columns=['Slope'])
save_path = os.path.join(save_to_path, "v2_slope_info.csv")
v2_slope_post_info.to_csv(save_path, index = False)

# V2 over all participants
mean_v2 = np.mean(v2_means)  
std_v2 = np.std(v2_means) 
v2_lower_ci, v2_upper_ci = credible_interval(mean_v2, std_v2)

# T-test over the v2_slope posterior means
t_statistic, p_value = ttest_1samp(v2_means, 0)

# Save the t_statistic and p_value to the specified directory
save_t_path = os.path.join(save_to_path, "t_test_results.txt")
with open(save_t_path, 'w') as file:
    file.write(f"T-statistic: {t_statistic}\n")
    file.write(f"p-value: {p_value}\n")

     
