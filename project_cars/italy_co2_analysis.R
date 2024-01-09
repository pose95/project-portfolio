setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(readr)
library(ggplot2)
library(tseries)


ds <- read.csv('data.csv')
ds_italy <- subset(ds, Entity=="Italy" )
#omit na value
ds_italy <- na.omit(ds_italy)
#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_italy_norm <- lapply(ds_italy[, 4:9], MinMax)

######### electric battery ##########
ts_IT_battery <- ts(ds_italy_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_IT_battery <- window(ts_IT_battery, start=2001, end=2016)

ts_IT_co2 <-ts(ds_italy_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_IT_co2<-window(ts_IT_co2, start=2001, end=2016)
acf(data_IT_co2)

plot(ts_IT_battery, type="l")
plot(ts_IT_co2, type="l")
plot(data_IT_battery, data_IT_co2)
plot(ds_italy_norm$battery_electric_number, ds_italy_norm$co2_per_km, type="l")

plot(ds_italy_norm$battery_electric_number, type="l")
lines(ds_italy_norm$co2_per_km, col=2)
grangertest(ts_IT_battery, ts_IT_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_IT_battery, trace=TRUE, xreg=data_IT_co2)
fit_value <- fitted(fit)
plot(data_IT_battery, type="l")
lines(fit_value, col=2)
#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_IT_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_IT_battery, start = 2017)
accuracy(for1,actual_data) 

########plugin##########

ts_IT_plugin <- ts(ds_italy_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_IT_plugin <- window(ts_IT_plugin, start=2001, end=2016)

plot(ts_IT_plugin, type="l")
plot(ts_IT_co2, type="l")
plot(data_IT_plugin, data_IT_co2)

grangertest(ts_IT_plugin, ts_IT_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_IT_plugin, trace=TRUE, xreg=data_IT_co2)
fit_value <- fitted(fit)
plot(data_IT_plugin, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_IT_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_IT_plugin, start = 2017)
accuracy(for1,actual_data) 

####################fullmild########################

ts_IT_fullmild <- ts(ds_italy_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_IT_fullmild <- window(ts_IT_fullmild, start=2001, end=2016)

plot(ts_IT_fullmild, type="l")
plot(ts_IT_co2, type="l")
plot(data_IT_fullmild, data_IT_co2)

grangertest(ts_IT_fullmild, ts_IT_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_IT_fullmild, trace=TRUE, xreg=data_IT_co2)
fit_value <- fitted(fit)
plot(data_IT_fullmild, type="l")
lines(fit_value, col=2)
accuracy(fit)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_IT_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_IT_fullmild, start = 2017)
accuracy(for1,actual_data)

##################petrol###################

ts_IT_petrol <- ts(ds_italy_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_IT_petrol <- window(ts_IT_petrol, start=2001, end=2016)

plot(ts_IT_petrol, type="l")
plot(ts_IT_co2, type="l")
plot(data_IT_petrol, data_IT_co2)

grangertest(ts_IT_petrol, ts_IT_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_IT_petrol, trace=TRUE, xreg=data_IT_co2)
fit_value <- fitted(fit)
plot(data_IT_petrol, type="l")
lines(fit_value, col=2)
#check the residual

checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_IT_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_IT_petrol, start = 2017)
accuracy(for1,actual_data)

##########################diesel##########################

ts_IT_diesel <- ts(ds_italy_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_IT_diesel <- window(ts_IT_diesel, start=2001, end=2016)

plot(ts_IT_diesel, type="l")
plot(ts_IT_co2, type="l")
plot(data_IT_diesel, data_IT_co2)

grangertest(ts_IT_diesel, ts_IT_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_IT_diesel, trace=TRUE, xreg=data_IT_co2)
fit_value <- fitted(fit)
plot(data_IT_diesel, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_IT_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_IT_diesel, start = 2017)
accuracy(for1,actual_data)
