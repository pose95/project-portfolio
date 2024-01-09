setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(readr)
library(ggplot2)
library(tseries)


ds <- read.csv('data.csv')
ds_france <- subset(ds, Entity=="France" )
#omit na value
ds_france <- na.omit(ds_france)
#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_france_norm <- lapply(ds_france[, 4:9], MinMax)

######### electric battery ##########
ts_FR_battery <- ts(ds_france_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_FR_battery <- window(ts_FR_battery, start=2001, end=2016)

ts_FR_co2 <-ts(ds_france_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_FR_co2<-window(ts_FR_co2, start=2001, end=2016)

plot(ts_FR_battery, type="l")
plot(ts_FR_co2, type="l")
plot(data_FR_battery, data_FR_co2)

grangertest(ts_FR_battery, ts_FR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_FR_battery, trace=TRUE, xreg=data_FR_co2)
fit_value <- fitted(fit)
plot(data_FR_battery, type="l")
lines(fit_value, col=2)


#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_FR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_battery, start = 2017)
accuracy(for1,actual_data) 

########plugin##########

ts_FR_plugin <- ts(ds_france_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_FR_plugin <- window(ts_FR_plugin, start=2001, end=2016)

plot(ts_FR_plugin, type="l")
plot(ts_FR_co2, type="l")
plot(data_FR_plugin, data_FR_co2)

grangertest(ts_FR_plugin, ts_FR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_FR_plugin, trace=TRUE, xreg=data_FR_co2)
fit_value <- fitted(fit)
plot(data_FR_plugin, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_FR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_plugin, start = 2017)
accuracy(for1,actual_data) 

####################fullmild########################

ts_FR_fullmild <- ts(ds_france_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_FR_fullmild <- window(ts_FR_fullmild, start=2001, end=2016)

plot(ts_FR_fullmild, type="l")
plot(ts_FR_co2, type="l")
plot(data_FR_fullmild, data_FR_co2)

grangertest(ts_FR_fullmild, ts_FR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_FR_fullmild, trace=TRUE, xreg=data_FR_co2)
fit_value <- fitted(fit)
plot(data_FR_fullmild, type="l")
lines(fit_value, col=2)
accuracy(fit)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_FR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_fullmild, start = 2017)
accuracy(for1,actual_data)

##################petrol###################

ts_FR_petrol <- ts(ds_france_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_FR_petrol <- window(ts_FR_petrol, start=2001, end=2016)

plot(ts_FR_petrol, type="l")
plot(ts_FR_co2, type="l")
plot(data_FR_petrol, data_FR_co2)

grangertest(ts_FR_petrol, ts_FR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_FR_petrol, trace=TRUE, xreg=data_FR_co2)
fit_value <- fitted(fit)
plot(data_FR_petrol, type="l")
lines(fit_value, col=2)
#check the residual

checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_FR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_petrol, start = 2017)
accuracy(for1,actual_data)

##########################diesel##########################

ts_FR_diesel <- ts(ds_france_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_FR_diesel <- window(ts_FR_diesel, start=2001, end=2016)

plot(ts_FR_diesel, type="l")
plot(ts_FR_co2, type="l")
plot(data_FR_diesel, data_FR_co2)

grangertest(ts_FR_diesel, ts_FR_co2, order=1)

#applied arima model and see the results
fit <- auto.arima(data_FR_diesel, trace=TRUE, xreg=data_FR_co2)
fit_value <- fitted(fit)
plot(data_FR_diesel, type="l")
lines(fit_value, col=2)

#check the residual
checkresiduals(fit)
mean(fit$residuals) 

###forecast
for1<- forecast(fit, xreg=data_FR_co2, h=5)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_diesel, start = 2017)
accuracy(for1,actual_data)

