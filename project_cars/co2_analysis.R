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
actual_data <- window (ts_FR_battery, start = 2017)
for1<- forecast(fit, xreg=data_FR_co2)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_battery, start = 2017)
x<- window(ts_FR_co2, start=2017)
accuracy(for1,actual_data, xreg=x) #RMSE(Test)=0.08881124
