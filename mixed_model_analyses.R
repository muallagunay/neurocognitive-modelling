

# House Keeping ----
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
rm(list=ls())
library(tidyverse)
library(lmerTest)
library(brms)
library(ggplot2)
library(grid)

# Load Data ----

#load the data
if(Sys.info()[["user"]]=="1111"){
  Data <- readbulk::read_bulk(paste0(getwd(), "/EEG Data Organised/"))
  Data <- Data[,1:8]
}else{
  Data <- list.files(path = "/Users/mualla/_", pattern = "*.csv") %>%
    map_df(~read_csv(file.path("/Users/mualla/_", .)))
}

# Defining the variables
names(Data) <- c("trial","cj","cor","rt","choice","confRT","Pe","Subject")

#Converting subject numericals into factors to group them
Data$Subject <- as.factor(Data$Subject)

# ---- Mixed Model Regression for Confidence ----
# Analysis
# 1- Linear mixed-effect model

# 1.1 - varying intercept but the same slope
Data$Pe_scaled = scale (Data$Pe)

conf.m1 <- lmer(cj ~ Pe_scaled + I(Pe_scaled^2) + (1 | Subject), data = Data, REML=T) 
summary(conf.m1)

# 1.2 - varying intercept + slope 
conf.m2 <- lmer(cj ~ Pe_scaled + I(Pe_scaled^2) + (Pe_scaled | Subject), data = Data, REML=T, lmerControl(optimizer="Nelder_Mead")) 
summary(conf.m2)

# 1.3 - model comparison through likelihood ratio tests
anova(conf.m1,conf.m2) # model2 gives best fit

# 1.4  - Assumptions Check 
# Linearity
plot(conf.m2, which = 1)  # Residuals vs Fitted plot

# Homoscedasticity
plot(conf.m2, which = 3)  # Residuals vs Fitted plot

# Normality
hist(resid(conf.m2))      # Histogram of residuals
qqnorm(resid(conf.m2))    # Q-Q plot

# (1) Linearity
# (2) Homogeneity of variance
# (3) Normality 
# (4) Multicollinearity - VIF

# ---- Mixed Model Regression for ConfRT ----
#Analysis
Data$conf_rt_log = log (Data$confRT)
Data$conf_rt_scaled = scale (Data$conf_rt_log, scale = FALSE)

# 1- Linear mixed-effect model: continuous outcome
# 1.1 - varying intercept but the same slope
confRT.m1 <- lmer(conf_rt_scaled ~ Pe_scaled + I(Pe^2) + (1 | Subject), data = Data)
summary(confRT.m1)

# 1.2 - varying the intercept + slope
confRT.m2 <- lmer(conf_rt_scaled ~ Pe_scaled + I(Pe_scaled^2) + (Pe_scaled | Subject),data = Data, REML=T,lmerControl(optimizer="Nelder_Mead"))
summary(confRT.m2)

# 1.3 - model comparison
anova(confRT.m1,confRT.m2) #model2 is best

# 1.4  - Assumptions Check 
# resid_panel(confRT.m2)

# (1) Linearity
# (2) Homogeneity of variance
# (3) Normality 
# (4) Multicollinearity - VIF

# Linearity & Homoscedasticity
plot(confRT.m2, which = 1)  # Residuals vs Fitted plot

# Normality
hist(resid(confRT.m2))      # Histogram of residuals
qqnorm(resid(confRT.m2))    # Q-Q plot

# ---- Plotting the relation ----
# Plot confidence 
data_conf <- data.frame(effects::effect('Pe_scaled',conf.m2)) # this creates the plot with 5 datapoitns. 
plot_conf <- ggplot(data_conf, aes(x = Pe_scaled, y = fit)) +
  geom_line(color = "deepskyblue") +  
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.3) + 
 # geom_point(data = Data, aes(x = Pe_scaled, y = cj), color = "blue", size = 2) +  # Add scatter points
  labs(x = "Standardized Pe Amplitude", y = "Confidence Level") +
  theme_minimal() +
  theme(axis.title = element_text(size = 15))

# Plot confRT
data_conf_rt <- data.frame(effects::effect('Pe_scaled',confRT.m2)) 
plot_conf_rt <- ggplot(data_conf_rt, aes(x = Pe_scaled, y = fit)) +
  # xlim(0, NA) + 
  # geom_line(color = "deepskyblue") + 
  # geom_ribbon(aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.3) + 
  geom_line(data = subset(data_conf_rt, Pe_scaled >= 0), color = "deepskyblue") +  
  geom_ribbon(data = subset(data_conf_rt, Pe_scaled >= 0), aes(ymin = lower, ymax = upper), fill = "lightblue", alpha = 0.3) + 
  # geom_point(data = Data, aes(x = Pe_scaled, y = conf_rt_scaled), color = "blue", size = 2) +  # Add scatter points
  scale_y_continuous(limits = c(-0.2 ,1.2)) + 
  labs(x = "Standardized Pe Amplitude", y = "ConfRT (s)") +
  theme_minimal() +
  theme(axis.title = element_text(size = 15))

plots <- grid.arrange(plot_conf, plot_conf_rt, ncol = 2)
ggsave("mixed_models_plots.pdf", plots, width = 12, height = 4)
