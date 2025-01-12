#This script fits the decision and confidence data from BY(2015) to Confidence DDM. 
#for Mualla's Master's Thesis 2022-2024

# House Keeping ----
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
rm(list=ls()) #empties the environment !!

library(Rcpp) # to source, compile and run C++ functions
library(DEoptim) # optimization algorithm
library(tidyverse)
library(BayesFactor)
sourceCpp("DDM_EEGconfidence_bounds.cpp") # this will give R access to the DDM_EEGconfidence_bounds function 

# Load data ----
if(Sys.info()[["user"]]=="1111"){
  Data <- readbulk::read_bulk(paste0(getwd(), "/EEG Data Organised/"))
  Data <- Data[,1:8]
}else{
  Data <- list.files(path = "/Users/mualla/_", pattern = "*.csv") %>%
    map_df(~read_csv(file.path("/Users/mualla/_", .)))
  }

names(Data) <- c("trial","cj","cor","rt","choice","rt2","Pe","Subject")
Data = subset(Data, select = -c(trial))
Data$cor <- 1-Data$cor #converting 0's into 1's so correct = 1, incorrect = 0.
Data$cj<-ifelse(Data$cj=="4"|Data$cj=="5"|Data$cj=="6",1,0) #confidence 4,5,6 = 1, confidence 1,2,3 = 0. 

# Standardize EEG data
eeg_mean <- mean(Data$Pe)
eeg_std <- sd(Data$Pe)
Data$stand_eeg <- (Data$Pe - eeg_mean ) / eeg_std

# Create EEG data bins ( 10 bins centered around 0)
Data$bins <-NA
for (i in 1:16){
Data$bins[Data$Subject==i] <-cut(Data$stand_eeg[Data$Subject==i], breaks = quantile(Data$stand_eeg[Data$Subject==i], probs = seq(0, 1, by = 0.1)), labels = FALSE) - 5.5
}

assign_bin_mean <- function(data_column, bins) {
  bin_means <- tapply(data_column, bins, mean)
  return(bin_means[bins])
}

#Creating a data frame to add parameter estimations 
est.params = data.frame(matrix(ncol = 8, nrow = 16))
names(est.params) <- c("v","a","ter","v2_intercept","v2_slope","a2","ter2","bestval")

# Define chi_square function ----
chi_square_optim <- function(params, Data, returnFit){
  z = .5 #starting point
  ntrials = length(Data$Subject) #1000 #ntrials
  s = 1 #within trial noise, fixed to 1
  
  # First, generate predictions:
  # With different parameters, as if we don't know the real ones)
  names(params) <- c('v','a','ter','v2_intercept','v2_slope','a2','ter2')
  predictions <- data.frame(DDM_EEGconfidence_bounds(v=params['v'],a=params['a'],ter=params['ter'],z=z,ntrials=ntrials,s=s,dt=.001,a2=params['a2'], Pe = Data$bins, v2_intercept=params['v2_intercept'],v2_slope=params['v2_slope'],a2_slope=0, ter2=params['ter2']))
  names(predictions) <- c('rt','resp','cor','raw_evidence2','rtfull','rt2','cj')
  
  # separate decision predictions according to the response
  c_predicted <- predictions[predictions$cor == 1,]
  e_predicted <- predictions[predictions$cor == 0,]
  
  # to make the next step easier, lets sort the predictions for correct and errors
  c_predicted_rt <- sort(c_predicted$rt) #RT for correct trials
  e_predicted_rt <- sort(e_predicted$rt) #RT for error trials 
  
  # separate confidence predictions according to the response (correct versus error)
  c_conf_predicted <- predictions[predictions$cor == 1,]
  e_conf_predicted <- predictions[predictions$cor == 0,]
  
  # to make the next step easier, lets sort the predictions for correct and errors
  c_conf_predicted_rt <- sort(c_predicted$rt2) #RT for correct trials
  e_conf_predicted_rt <- sort(e_predicted$rt2) #RT for error trials
  
  # again, separate confidence predictions according to the response
  h_predicted <- predictions[predictions$cj == 1,]
  l_predicted <- predictions[predictions$cj == 0,]
  
  # sort confidence predictions for high and low
  h_predicted_rt <- sort(h_predicted$rt2) #RT for high confidence trials
  l_predicted_rt <- sort(l_predicted$rt2) #RT for low confidence trials 
  
  #1 Decision
  #if we're only simulating data, return the predictions
  if(returnFit==0){ 
    return(predictions[,c('rt','cor','rt2','cj')])
    
    #If we are fitting the model, now compare these predictions to the observations 
  }else{ 
    
    # First, separate the data in correct and error trials
    c_observed <- Data[Data$cor == 1,]
    e_observed <- Data[Data$cor == 0,]
    
    # Now, get the quantile RTs on the "observed data" for correct and error distributions separately (for quantiles .1, .3, .5, .7, .9)
    c_quantiles <- quantile(c_observed$rt, probs = c(.1,.3,.5,.7,.9), names = FALSE) #we are looking at: how much of the data falls into these quantiles? 
    e_quantiles <- quantile(e_observed$rt, probs = c(.1,.3,.5,.7,.9), names = FALSE)
    
    # to combine correct and incorrect we scale the expected interquantile probability by the proportion of correct and incorrect respectively
    prop_obs_c <- dim(c_observed)[1] / dim(Data)[1]
    prop_obs_e <- dim(e_observed)[1] / dim(Data)[1]
    
    c_obs_proportion = prop_obs_c * c(.1, .2, .2, .2, .2, .1)
    e_obs_proportion = prop_obs_e * c(.1, .2, .2, .2, .2, .1)
    obs_props <- c(c_obs_proportion,e_obs_proportion)
    
    # now, get the proportion of responses that fall between the observed quantiles when applied to the predicted data (scale by N?)
    c_pred_proportion <- c(
      sum(c_predicted_rt <= c_quantiles[1]), 
      sum(c_predicted_rt <= c_quantiles[2]) - sum(c_predicted_rt <= c_quantiles[1]),
      sum(c_predicted_rt <= c_quantiles[3]) - sum(c_predicted_rt <= c_quantiles[2]),
      sum(c_predicted_rt <= c_quantiles[4]) - sum(c_predicted_rt <= c_quantiles[3]),
      sum(c_predicted_rt <= c_quantiles[5]) - sum(c_predicted_rt <= c_quantiles[4]),
      sum(c_predicted_rt > c_quantiles[5])
    ) / dim(predictions)[1]
    
    e_pred_proportion <- c(
      sum(e_predicted_rt <= e_quantiles[1]),
      sum(e_predicted_rt <= e_quantiles[2]) - sum(e_predicted_rt <= e_quantiles[1]),
      sum(e_predicted_rt <= e_quantiles[3]) - sum(e_predicted_rt <= e_quantiles[2]),
      sum(e_predicted_rt <= e_quantiles[4]) - sum(e_predicted_rt <= e_quantiles[3]),
      sum(e_predicted_rt <= e_quantiles[5]) - sum(e_predicted_rt <= e_quantiles[4]),
      sum(e_predicted_rt > e_quantiles[5])
    ) / dim(predictions)[1]
    pred_props <- c(c_pred_proportion,e_pred_proportion)
    
    # avoid zeros in the the data (because of division by predictions for chi square statistic) -> set to small number
    pred_props[pred_props==0] <- .0000001
    
    #2 Confidence in correct and error trials 
    
    # First, separate the data in correct and error trials #add based on correct and error, instead of high and low
    c_conf_observed <- Data[Data$cor == 1,]
    e_conf_observed <- Data[Data$cor == 0,]
    
    # Now, get the quantile RTs on the "observed data" for correct and error distributions separately (for quantiles .1, .3, .5, .7, .9)
    c_conf_quantiles <- quantile(c_conf_observed$rt2, probs = c(.1,.3,.5,.7,.9), names = FALSE) #we are looking at: how much of the data falls into these quantiles? 
    e_conf_quantiles <- quantile(e_conf_observed$rt2, probs = c(.1,.3,.5,.7,.9), names = FALSE)
    
    # to combine correct and incorrect we scale the expected interquantile probability by the proportion of correct and incorrect respectively
    prop_obs_c_conf <- dim(c_conf_observed)[1] / dim(Data)[1]
    prop_obs_e_conf <- dim(e_conf_observed)[1] / dim(Data)[1]
    
    c_conf_obs_proportion = prop_obs_c_conf * c(.1, .2, .2, .2, .2, .1)
    e_conf_obs_proportion = prop_obs_e_conf * c(.1, .2, .2, .2, .2, .1)
    ce_conf_obs_props <- c(c_conf_obs_proportion,e_conf_obs_proportion)
    
    # now, get the proportion of responses that fall between the observed quantiles when applied to the predicted data (scale by N?)
    c_conf_pred_proportion <- c(
      sum(c_conf_predicted_rt <= c_conf_quantiles[1]), 
      sum(c_conf_predicted_rt <= c_conf_quantiles[2]) - sum(c_conf_predicted_rt <= c_conf_quantiles[1]),
      sum(c_conf_predicted_rt <= c_conf_quantiles[3]) - sum(c_conf_predicted_rt <= c_conf_quantiles[2]),
      sum(c_conf_predicted_rt <= c_conf_quantiles[4]) - sum(c_conf_predicted_rt <= c_conf_quantiles[3]),
      sum(c_conf_predicted_rt <= c_conf_quantiles[5]) - sum(c_conf_predicted_rt <= c_conf_quantiles[4]),
      sum(c_conf_predicted_rt > c_conf_quantiles[5])
    ) / dim(predictions)[1]
    
    e_conf_pred_proportion <- c(
      sum(e_conf_predicted_rt <= e_conf_quantiles[1]),
      sum(e_conf_predicted_rt <= e_conf_quantiles[2]) - sum(e_conf_predicted_rt <= e_conf_quantiles[1]),
      sum(e_conf_predicted_rt <= e_conf_quantiles[3]) - sum(e_conf_predicted_rt <= e_conf_quantiles[2]),
      sum(e_conf_predicted_rt <= e_conf_quantiles[4]) - sum(e_conf_predicted_rt <= e_conf_quantiles[3]),
      sum(e_conf_predicted_rt <= e_conf_quantiles[5]) - sum(e_conf_predicted_rt <= e_conf_quantiles[4]),
      sum(e_conf_predicted_rt > e_conf_quantiles[5])
    ) / dim(predictions)[1]
    ce_conf_pred_props <- c(c_conf_pred_proportion,e_conf_pred_proportion)
    
    # avoid zeros in the the data (because of division by predictions for chi square statistic) -> set to small number
    ce_conf_pred_props[ce_conf_pred_props==0] <- .0000001
    
    #3 Confidence in high and low trials 
    
    # First, separate the data in high and low trials
    
    h_observed <- Data[Data$cj == 1,]
    l_observed <- Data[Data$cj == 0,]
    
    # Now, get the quantile RTs on the "observed data" for correct and error distributions separately (for quantiles .1, .3, .5, .7, .9)
    h_quantiles <- quantile(h_observed$rt2, probs = c(.1,.3,.5,.7,.9), names = FALSE) #we are looking at: how much of the data falls into these quantiles? 
    l_quantiles <- quantile(l_observed$rt2, probs = c(.1,.3,.5,.7,.9), names = FALSE)
    
    # to combine correct and incorrect we scale the expected interquantile probability by the proportion of correct and incorrect respectively
    prop_obs_h <- dim(h_observed)[1] / dim(Data)[1]
    prop_obs_l <- dim(l_observed)[1] / dim(Data)[1]
    
    h_obs_proportion = prop_obs_h * c(.1, .2, .2, .2, .2, .1)
    l_obs_proportion = prop_obs_l * c(.1, .2, .2, .2, .2, .1)
    conf_obs_props <- c(h_obs_proportion,l_obs_proportion)
    
    # now, get the proportion of responses that fall between the observed quantiles when applied to the predicted data (scale by N?)
    h_pred_proportion <- c(
      sum(h_predicted_rt <= h_quantiles[1]), 
      sum(h_predicted_rt <= h_quantiles[2]) - sum(h_predicted_rt <= h_quantiles[1]),
      sum(h_predicted_rt <= h_quantiles[3]) - sum(h_predicted_rt <= h_quantiles[2]),
      sum(h_predicted_rt <= h_quantiles[4]) - sum(h_predicted_rt <= h_quantiles[3]),
      sum(h_predicted_rt <= h_quantiles[5]) - sum(h_predicted_rt <= h_quantiles[4]),
      sum(h_predicted_rt > h_quantiles[5])
    ) / dim(predictions)[1]
    
    l_pred_proportion <- c(
      sum(l_predicted_rt <= l_quantiles[1]),
      sum(l_predicted_rt <= l_quantiles[2]) - sum(l_predicted_rt <= l_quantiles[1]),
      sum(l_predicted_rt <= l_quantiles[3]) - sum(l_predicted_rt <= l_quantiles[2]),
      sum(l_predicted_rt <= l_quantiles[4]) - sum(l_predicted_rt <= l_quantiles[3]),
      sum(l_predicted_rt <= l_quantiles[5]) - sum(l_predicted_rt <= l_quantiles[4]),
      sum(l_predicted_rt > l_quantiles[5])
    ) / dim(predictions)[1]
    conf_pred_props <- c(h_pred_proportion,l_pred_proportion)
    
    # avoid zeros in the the data (because of division by predictions for chi square statistic) -> set to small number
    conf_pred_props[conf_pred_props==0] <- .0000001
    
    #Combining the quantiles for decision and confidence
    obs_props <- c(obs_props, obs_props, ce_conf_obs_props, conf_obs_props) 
    pred_props <- c(pred_props, pred_props, ce_conf_pred_props,conf_pred_props)
    
    #chiSquare
    chiSquare = sum( ( (obs_props - pred_props) ^ 2) / pred_props )
    
    #Return chiSquare
    return(chiSquare)
  }
}

subs <- unique(Data$Subject) ; N <- unique(subs)

# Parameter estimation ----
# Estimate parameters based on observed data

# v = decision drift rate
# a = decision boundary
# ter = non-decision time
# v2_slope = confidence drift rate slope
# v2_intercept = confidence drift rate intercept
# a2 = confidence boundary
# ter2 =confidence non-decision time

for (i in 1:16) {
  #Load existing individual results if already exist
    if(file.exists(paste0('/Users/mualla/_/results_sub_',i,'.Rdata'))){
    load(paste0('/Users/mualla/_/results_sub_',i,'.Rdata'))
    #Save the parameters
    est.params$v[i] <- (results$optim$bestmem[1])
    est.params$a[i] <- (results$optim$bestmem[2])
    est.params$ter[i] <- (results$optim$bestmem[3])
    est.params$v2_intercept[i] <- (results$optim$bestmem[4])
    est.params$v2_slope[i] <- (results$optim$bestmem[5])
    est.params$a2[i] <- (results$optim$bestmem[6])
    est.params$ter2[i] <- (results$optim$bestmem[7])
    est.params$bestval[i] <-(results$optim$bestval)
    }else{ #if not, fit the model
    par(mfrow = c(1, 2))
      
    hist(Data$rt[Data$cor == 1 & Data$Subject==i],col="green",xlab="Reaction times (s)",main = paste("Decision Participant #", i)) #plotting observations for correct trials
    hist(Data$rt[Data$cor == 0 & Data$Subject==i],add=T,col="red") #plotting observations for incorrect trials
    
    hist(Data$rt2[Data$cj == 1 & Data$Subject==i],col="green",xlab="Reaction times (s)",main = paste("Confidence Participant #", i)) #plotting observations for correct trials
    hist(Data$rt2[Data$cj == 0 & Data$Subject==i],add=T,col="red") #plotting observations for incorrect trials
    
    optimal_params <- DEoptim(
      chi_square_optim, # function to optimize
      lower = c(0, .5, 0, 0, -5, .5, -1), # v, a, ter, v2_slope, v2_intercept, a2, ter2  
      upper = c(3, 4, 1,  3, 5,  4, 1),
      
      Data = Data[Data$Subject==i,],
      returnFit = 1,
      control = c(itermax = 1000, steptol=100, reltol=.001, NP=70) #the iterations stop if the cost function doesn't change for 100 iterations
    )
    results <- summary(optimal_params) #These parameter values should be similar to those above
    #save individual results
    save(results, file=paste0('/Users/mualla/_/results_sub_',i,'.Rdata'))
  }
}

# Plots ----
# Plot the empirical data with the simulated data (simulated based on estimated parameters)
par(mfrow = c(1, 2))
for (i in 1:16) {
  Simuls <- chi_square_optim(params = as.numeric(est.params[i,]), Data = Data[Data$Subject==i,],returnFit=0)
  Simuls$sub = i
  if(!exists('All_Simuls')){ All_Simuls <- Simuls
  }else{ All_Simuls <- rbind(All_Simuls,Simuls)
  
 }
  
  #Compare empirical and simulated data
  #bars show the empirical data, lines show the simulated data
  tempC <- hist(Data$rt[Data$cor==1 & Data$Subject==i],breaks=seq(0,3,.05),xlim=c(0,2.5),prob=F,col=rgb(0,1,0,.25),border="white",ylab="Frequency",xlab="Reaction times (s)",cex.lab=2, cex.main=1.5, cex.axis=1.5,main = paste("Decision Plot Participant #", i))
  tempE <- hist(Data$rt[Data$cor==0 & Data$Subject==i],breaks=seq(0,3,.05),prob=F,add=T,col=rgb(1,0,0,.25),border='white')
  Cors <- hist(Simuls$rt[Simuls$cor==1],breaks=seq(0,20,.05),plot=F)
  Errs <- hist(abs(Simuls$rt[Simuls$cor==0]),breaks=seq(0,20,.05),plot=F)
  lines(Cors$counts/(sum(Cors$counts)/sum(tempC$counts))~Cors$mids,type='l',col='green',lwd=3)
  lines(Errs$counts/(sum(Errs$counts)/sum(tempE$counts))~Errs$mids,type='l',col='red',lwd=3)
  print (paste0('Decision plot participant #' , i))
  
  #Compare empirical and simulated data
  #bars show the empirical data, lines show the simulated data
  tempH <- hist(Data$rt2[Data$cj==1 & Data$Subject==i],breaks=seq(0,15,.05),xlim=c(0,2.5),prob=F,col=rgb(0,1,0,.25),border="white",ylab="Frequency",xlab="Reaction times (s)",cex.lab=2, cex.main=1.5, cex.axis=1.5,main = paste("Confidence Plot Participant #", i))
  tempL <- hist(Data$rt2[Data$cj==0 & Data$Subject==i],breaks=seq(0,30,.05),prob=F,add=T,col=rgb(1,0,0,.25),border='white')
  High <- hist(Simuls$rt2[Simuls$cj==1],breaks=seq(-1,50,.05), plot=F)
  Low <- hist(abs(Simuls$rt2[Simuls$cj==0]),breaks=seq(-1,50,.05),plot=F)
  lines(High$counts/(sum(High$counts)/sum(tempH$counts))~High$mids,type='l',col='green',lwd=3)
  lines(Low$counts/(sum(Low$counts)/sum(tempL$counts))~Low$mids,type='l',col='red',lwd=3)
  print (paste0('Confidence plot participant #' , i))
}

# Create plot for all subjects
par(mfrow = c(1, 2))
tempC <- hist(Data$rt[Data$cor==1],breaks=seq(0,3,.05),xlim=c(0,2.5),prob=F,col=rgb(0,1,0,.25),border="white",ylab="Frequency",xlab="Decision RT (s)",cex.lab=1.2, cex.main=1.5, cex.axis=1.2,main = paste(""))
tempE <- hist(Data$rt[Data$cor==0],breaks=seq(0,3,.05),prob=F,add=T,col=rgb(1,0,0,.25),border='white')
Cors <- hist(All_Simuls$rt[All_Simuls$cor==1],breaks=seq(0,20,.05),plot=F)
Errs <- hist(abs(All_Simuls$rt[All_Simuls$cor==0]),breaks=seq(0,20,.05),plot=F)
lines(Cors$counts/(sum(Cors$counts)/sum(tempC$counts))~Cors$mids,type='l',col='green',lwd=3)
lines(Errs$counts/(sum(Errs$counts)/sum(tempE$counts))~Errs$mids,type='l',col='red',lwd=3)
legend("topright", legend = c("Correct Choice", "Error Choice"), col = c("green", "red"), bty = "l", pt.cex = 1.5, pch = 15, cex = 0.8)

tempH <- hist(Data$rt2[Data$cj==1],breaks=seq(0,15,.05),xlim=c(0,1.5),prob=F,col=rgb(0,1,0,.25),border="white",ylab="Frequency",xlab=" ConfRT (s)",cex.lab=1.2, cex.main=1.5, cex.axis=1.2,main = paste(""))
tempL <- hist(Data$rt2[Data$cj==0],breaks=seq(0,30,.05),prob=F,add=T,col=rgb(1,0,0,.25),border='white')
High <- hist(All_Simuls$rt2[All_Simuls$cj==1],breaks=seq(-1,50,.05), plot=F)
Low <- hist(abs(All_Simuls$rt2[All_Simuls$cj==0]),breaks=seq(-1,50,.05),plot=F)
lines(High$counts/(sum(High$counts)/sum(tempH$counts))~High$mids,type='l',col='green',lwd=3)
lines(Low$counts/(sum(Low$counts)/sum(tempL$counts))~Low$mids,type='l',col='red',lwd=3)
legend("topright", legend = c("High Confidence", "Low Confidence"), col = c("green", "red"), bty = "l", pt.cex = 1.5, pch = 15, cex = 0.8)

# est.params <- cbind(Subject = 1:16, est.params)
# write.csv(est.params, file = "/Users/mualla/_/estimated_params.csv", row.names = FALSE)

# Significance test ----

#Frequentist: Does the v2_slope differ from 0? 
t.test(est.params$v2_slope)

# Bayesian: Does the v2_slope differ from 0? 
1 / ttestBF(x = est.params.bins$v2_slope, mu =0)

slope <- est.params.bins$v2_slope

# Calculate mean of each column
means <- colMeans(est.params.bins)

# Calculate standard deviation of each column
sds <- apply(est.params.bins, 2, sd)




