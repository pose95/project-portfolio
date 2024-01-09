setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(readr)
library(ggplot2)
library(tseries)

ds <- read.csv('data.csv')
ds_nd <- subset(ds, Entity=="Netherlands" )
#omit na value
ds_nd <- na.omit(ds_nd)
#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_nd_norm <- lapply(ds_nd[, 4:9], MinMax)

######### electric battery ##########
ts_nd_battery <- ts(ds_nd_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_nd_battery <- window(ts_nd_battery, start=2001, end=2016)

ts_nd_co2 <-ts(ds_nd_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_nd_co2<-window(ts_nd_co2, start=2001, end=2016)

plot(ts_nd_battery, type="l")
plot(ts_nd_co2, type="l")
plot(data_nd_battery, data_nd_co2)

grangertest(ts_nd_battery, ts_nd_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_nd_battery, trace=TRUE, xreg=data_nd_co2)
fit_value <- fitted(fit)
plot(data_nd_battery, type="l")
lines(fit_value, col=2)


#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_nd_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_nd_battery, start = 2017)
accuracy(for1,actual_data) 

########plugin##########

ts_nd_plugin <- ts(ds_nd_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_nd_plugin <- window(ts_nd_plugin, start=2001, end=2016)

plot(ts_nd_plugin, type="l")
plot(ts_nd_co2, type="l")
plot(data_nd_plugin, data_nd_co2)

grangertest(ts_nd_plugin, ts_nd_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_nd_plugin, trace=TRUE, xreg=data_nd_co2)
fit_value <- fitted(fit)
plot(data_nd_plugin, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_nd_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_nd_plugin, start = 2017)
accuracy(for1,actual_data) 

####################fullmild########################

ts_nd_fullmild <- ts(ds_nd_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_nd_fullmild <- window(ts_nd_fullmild, start=2001, end=2016)

plot(ts_nd_fullmild, type="l")
plot(ts_nd_co2, type="l")
plot(data_nd_fullmild, data_nd_co2)

grangertest(ts_nd_fullmild, ts_nd_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_nd_fullmild, trace=TRUE, xreg=data_nd_co2)
fit_value <- fitted(fit)
plot(data_nd_fullmild, type="l")
lines(fit_value, col=2)
accuracy(fit)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_nd_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_nd_fullmild, start = 2017)
accuracy(for1,actual_data)

##################petrol###################

ts_nd_petrol <- ts(ds_nd_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_nd_petrol <- window(ts_nd_petrol, start=2001, end=2016)

plot(ts_nd_petrol, type="l")
plot(ts_nd_co2, type="l")
plot(data_nd_petrol, data_nd_co2)

grangertest(ts_nd_petrol, ts_nd_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_nd_petrol, trace=TRUE, xreg=data_nd_co2)
fit_value <- fitted(fit)
plot(data_nd_petrol, type="l")
lines(fit_value, col=2)
#check the residual

checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_nd_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_nd_petrol, start = 2017)
accuracy(for1,actual_data)

##########################diesel##########################

ts_nd_diesel <- ts(ds_nd_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_nd_diesel <- window(ts_nd_diesel, start=2001, end=2016)

plot(ts_nd_diesel, type="l")
plot(ts_nd_co2, type="l")
plot(data_nd_diesel, data_nd_co2)

grangertest(ts_nd_diesel, ts_nd_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_nd_diesel, trace=TRUE, xreg=data_nd_co2)
fit_value <- fitted(fit)
plot(data_nd_diesel, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_nd_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_nd_diesel, start = 2017)
accuracy(for1,actual_data)

