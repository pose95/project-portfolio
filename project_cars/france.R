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
head(ds_france_norm)

######electric battery cars###########

#see if there are outliers
par(bty="l")
boxplot(ds_france$battery_electric_number, main="electric battery")

ts_FR_battery <- ts(ds_france_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_FR_battery <- window(ts_FR_battery, start=2001, end=2016)

#apply the differenciation time 
elect_batt_diff1 <- diff(data_FR_battery, differences=2)
plot(ds_france$Year[1:14], elect_batt_diff1, type="l")

adf.test(elect_batt_diff1)

## we fit a linear model with the tslm function
fitts1<- tslm(data_FR_battery~trend)

plot(data_FR_battery, type="l")
lines(fitted(fitts1), col=3)

##obviously it gives the same results of the first model
summary(fitts1)

dwtest(fitts1)

#check the residuals
resfitts1 <- residuals(fitts1)
plot(resfitts1,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts1)

forecast1 <- forecast(fitts1, h=5)

plot(forecast1)

AIC(fitts1)

### Check forecast against actuals
actual_data2 <- window (ts_FR_battery, start = 2017)
accuracy(forecast1, actual_data2) #RMSE(Test)=0.05997754

######plugin cars######
#check outliers
par(bty="l")
boxplot(ds_france$plugin_hybrid_number, main="plugin hybrid number")

ts_FR_plugin <- ts(ds_france_norm$plugin_hybrid_number, start =2001, end=2019, frequency = 1)
data_FR_plugin <- window(ts_FR_plugin, start=2001, end=2016)

#apply the differenciation time 
plugin_diff1 <- diff(data_FR_plugin, differences=1)
plot(ds_france$Year[1:15], plugin_diff1, type="l")

adf.test(plugin_diff1)

## we fit a linear model with the tslm function
fitts2<- tslm(plugin_diff1~trend)

plot(plugin_diff1, type="l")
lines(fitted(fitts2), col=3)

##obviously it gives the same results of the first model
summary(fitts2)

dwtest(fitts2)

#check the residuals
resfitts2 <- residuals(fitts2)
plot(resfitts2,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts2)

forecast2 <- forecast(fitts2, h=5)
plot(forecast2)

AIC(fitts2)

### Check forecast against actuals
actual_data2 <- window (ts_FR_plugin, start = 2017)
accuracy(forecast2, actual_data2) #RMSE(Test)=0.05997754

######## full mild hybrid ########
#check the outliers
par(bty="l")
boxplot(ds_france$full_mild_hybrid_number, main="fullmild hybrid number")

ts_FR_fullmild <- ts(ds_france_norm$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)
data_FR_fullmild<- window(ts_FR_fullmild, start=2001, end=2016)

#apply the differenciation time 
fullmild_diff1 <- diff(data_FR_fullmild, differences=1)
plot(ds_france$Year[1:18], fullmild_diff1, type="l")

adf.test(fullmild_diff1)

## we fit a linear model with the tslm function
fitts2<- tslm(fullmild_diff1~trend)

plot(fullmild_diff1, type="l")
lines(fitted(fitts2), col=3)

##obviously it gives the same results of the first model
summary(fitts2)

dwtest(fitts2)

#check the residuals
resfitts2 <- residuals(fitts2)
plot(resfitts2,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts2)

forecast2 <- forecast(fitts2, h=5)
plot(forecast2)

AIC(fitts2)

#check forecast against actual
actual_data2 <- window (ts_FR_fullmild, start = 2017)
accuracy(forecast2, actual_data2) #RMSE(Test)=0.05997754

############ petrol cars###########
#check outliers
par(bty="l")
boxplot(ds_france$petrol_number, main="petrol number")

ts_FR_petrol <- ts(ds_france_norm$petrol_number, start =2001, end=2019, frequency = 1)
data_FR_petrol<- window(ts_FR_petrol, start=2001, end=2016)

#apply the differenciation time 
petrol_diff1 <- diff(ts_FR_petrol, differences=1)
plot(ds_france$Year[1:18], petrol_diff1, type="l")

adf.test(petrol_diff1)


## we fit a linear model with the tslm function
fitts2<- tslm(petrol_diff1~trend)

plot(petrol.ts2, type="l")
lines(fitted(fitts2), col=3)

##obviously it gives the same results of the first model
summary(fitts2)

dwtest(fitts2)

#check the residuals
resfitts2 <- residuals(fitts2)
plot(resfitts2,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts2)

forecast2 <- forecast(fitts2, h=5)
plot(forecast2)

AIC(fitts2)

#check forecast against actual
actual_data2 <- window (ts_FR_petrol, start = 2017)
accuracy(forecast2, actual_data2) #RMSE(Test)=0.05997754

###### diesel cars #########
#check outliers
par(bty="l")
boxplot(ds_france$diesel_gas_number, main="diesel number")

ts_FR_diesel <- ts(ds_france_norm$diesel_gas_number, start =2001, end=2019, frequency = 1)
data_FR_diesel<- window(ts_FR_diesel, start=2001, end=2016)

#apply the differenciation time 
diesel_diff1 <- diff(data_FR_diesel, differences=1)
plot(ds_france$Year[1:18], diesel_diff1, type="l")

adf.test(diesel_diff1)

## we fit a linear model with the tslm function
fitts2<- tslm(diesel_diff1~trend)

plot(diesel_diff1, type="l")
lines(fitted(fitts2), col=3)

##obviously it gives the same results of the first model
summary(fitts2)

dwtest(fitts2)

#check the residuals
resfitts2 <- residuals(fitts2)
plot(resfitts2,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts2)

forecast2 <- forecast(fitts2, h=5)
plot(forecast2)

AIC(fitts2)

#check forecast against actual
actual_data2 <- window (ts_FR_diesel, start = 2017)
accuracy(forecast2, actual_data2) #RMSE(Test)=0.05997754

#####co2 emissions##########
par(bty="l")
boxplot(ds_france$co2_per_km, main="co2 emissions")

ts_FR_co2 <- ts(ds_france_norm$co2_per_km, start =2001, end=2019, frequency = 1)
data_FR_co2<- window(ts_FR_co2, start=2001, end=2016)

#apply the differenciation time 
co2_diff1 <- diff(data_FR_diesel, differences=1)
plot(ds_france$Year[1:18], co2_diff1, type="l")

adf.test(diesel_diff1)

## we fit a linear model with the tslm function
fitts2<- tslm(co2_diff1~trend)

plot(co2_diff1, type="l")
lines(fitted(fitts2), col=3)

##obviously it gives the same results of the first model
summary(fitts2)

dwtest(fitts2)

#check the residuals
resfitts2 <- residuals(fitts2)
plot(resfitts2,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function on the residuals
acf(resfitts2)

forecast2 <- forecast(fitts2, h=5)
plot(forecast2)

AIC(fitts2)

#check forecast against actual
actual_data2 <- window (ts_FR_co2, start = 2017)
accuracy(forecast2, actual_data2) #RMSE(Test)=0.05997754
