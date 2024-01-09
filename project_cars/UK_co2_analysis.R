setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(readr)
library(ggplot2)
library(tseries)


ds <- read.csv('data.csv')
ds_uk <- subset(ds, Entity=="United Kingdom" )
#omit na value
ds_uk <- na.omit(ds_uk)
#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_uk_norm <- lapply(ds_uk[, 4:9], MinMax)

######### electric battery ##########
ts_uk_battery <- ts(ds_uk_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_uk_battery <- window(ts_uk_battery, start=2001, end=2016)

ts_uk_co2 <-ts(ds_uk_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_uk_co2<-window(ts_uk_co2, start=2001, end=2016)

plot(ts_uk_battery, type="l")
plot(ts_uk_co2, type="l")
plot(data_uk_battery, data_uk_co2)

grangertest(ts_uk_battery, ts_uk_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_uk_battery, trace=TRUE, xreg=data_uk_co2)
fit_value <- fitted(fit)
plot(data_uk_battery, type="l")
lines(fit_value, col=2)


#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_uk_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_uk_battery, start = 2017)
accuracy(for1,actual_data) 

########plugin##########

ts_uk_plugin <- ts(ds_uk_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_uk_plugin <- window(ts_uk_plugin, start=2001, end=2016)

plot(ts_uk_plugin, type="l")
plot(ts_uk_co2, type="l")
plot(data_uk_plugin, data_uk_co2)

grangertest(ts_uk_plugin, ts_uk_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_uk_plugin, trace=TRUE, xreg=data_uk_co2)
fit_value <- fitted(fit)
plot(data_uk_plugin, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_uk_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_uk_plugin, start = 2017)
accuracy(for1,actual_data) 

####################fullmild########################

ts_uk_fullmild <- ts(ds_uk_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_uk_fullmild <- window(ts_uk_fullmild, start=2001, end=2016)

plot(ts_uk_fullmild, type="l")
plot(ts_uk_co2, type="l")
plot(data_uk_fullmild, data_uk_co2)

grangertest(ts_uk_fullmild, ts_uk_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_uk_fullmild, trace=TRUE, xreg=data_uk_co2)
fit_value <- fitted(fit)
plot(data_uk_fullmild, type="l")
lines(fit_value, col=2)
accuracy(fit)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_uk_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_uk_fullmild, start = 2017)
accuracy(for1,actual_data)

##################petrol###################

ts_uk_petrol <- ts(ds_uk_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_uk_petrol <- window(ts_uk_petrol, start=2001, end=2016)

plot(ts_uk_petrol, type="l")
plot(ts_uk_co2, type="l")
plot(data_uk_petrol, data_uk_co2)

grangertest(ts_uk_petrol, ts_uk_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_uk_petrol, trace=TRUE, xreg=data_uk_co2)
fit_value <- fitted(fit)
plot(data_uk_petrol, type="l")
lines(fit_value, col=2)
#check the residual

checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_uk_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_uk_petrol, start = 2017)
accuracy(for1,actual_data)

##########################diesel##########################

ts_uk_diesel <- ts(ds_uk_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_uk_diesel <- window(ts_uk_diesel, start=2001, end=2016)

plot(ts_uk_diesel, type="l")
plot(ts_uk_co2, type="l")
plot(data_uk_diesel, data_uk_co2)

grangertest(ts_uk_diesel, ts_uk_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_uk_diesel, trace=TRUE, xreg=data_uk_co2)
fit_value <- fitted(fit)
plot(data_uk_diesel, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_uk_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_uk_diesel, start = 2017)
accuracy(for1,actual_data)

