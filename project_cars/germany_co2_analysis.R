setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(readr)
library(ggplot2)
library(tseries)


ds <- read.csv('data.csv')
ds_germany <- subset(ds, Entity=="Germany" )
#omit na value
ds_germany <- na.omit(ds_germany)
#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_germany_norm <- lapply(ds_germany[, 4:9], MinMax)

######### electric battery ##########
ts_GR_battery <- ts(ds_germany_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_GR_battery <- window(ts_GR_battery, start=2001, end=2016)

ts_GR_co2 <-ts(ds_germany_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_GR_co2<-window(ts_GR_co2, start=2001, end=2016)

plot(ts_GR_battery, type="l")
plot(ts_GR_co2, type="l")
plot(data_GR_battery, data_GR_co2)

grangertest(ts_GR_battery, ts_GR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_GR_battery, trace=TRUE, xreg=data_GR_co2)
fit_value <- fitted(fit)
plot(data_GR_battery, type="l")
lines(fit_value, col=2)


#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_GR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_GR_battery, start = 2017)
accuracy(for1,actual_data) 

########plugin##########

ts_GR_plugin <- ts(ds_germany_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_GR_plugin <- window(ts_GR_plugin, start=2001, end=2016)

plot(ts_GR_plugin, type="l")
plot(ts_GR_co2, type="l")
plot(data_GR_plugin, data_GR_co2)

grangertest(ts_GR_plugin, ts_GR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_GR_plugin, trace=TRUE, xreg=data_GR_co2)
fit_value <- fitted(fit)
plot(data_GR_plugin, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_GR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_GR_plugin, start = 2017)
accuracy(for1,actual_data) 

####################fullmild########################

ts_GR_fullmild <- ts(ds_germany_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_GR_fullmild <- window(ts_GR_fullmild, start=2001, end=2016)

plot(ts_GR_fullmild, type="l")
plot(ts_GR_co2, type="l")
plot(data_GR_fullmild, data_GR_co2)

grangertest(ts_GR_fullmild, ts_GR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_GR_fullmild, trace=TRUE, xreg=data_GR_co2)
fit_value <- fitted(fit)
plot(data_GR_fullmild, type="l")
lines(fit_value, col=2)
accuracy(fit)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_GR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_GR_fullmild, start = 2017)
accuracy(for1,actual_data)

##################petrol###################

ts_GR_petrol <- ts(ds_germany_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_GR_petrol <- window(ts_GR_petrol, start=2001, end=2016)

plot(ts_GR_petrol, type="l")
plot(ts_GR_co2, type="l")
plot(data_GR_petrol, data_GR_co2)

grangertest(ts_GR_petrol, ts_GR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_GR_petrol, trace=TRUE, xreg=data_GR_co2)
fit_value <- fitted(fit)
plot(data_GR_petrol, type="l")
lines(fit_value, col=2)
#check the residual

checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_GR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_GR_petrol, start = 2017)
accuracy(for1,actual_data)

##########################diesel##########################

ts_GR_diesel <- ts(ds_germany_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_GR_diesel <- window(ts_GR_diesel, start=2001, end=2016)

plot(ts_GR_diesel, type="l")
plot(ts_GR_co2, type="l")
plot(data_GR_diesel, data_GR_co2)

grangertest(ts_GR_diesel, ts_GR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_GR_diesel, trace=TRUE, xreg=data_GR_co2)
fit_value <- fitted(fit)
plot(data_GR_diesel, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_GR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_GR_diesel, start = 2017)
accuracy(for1,actual_data)

