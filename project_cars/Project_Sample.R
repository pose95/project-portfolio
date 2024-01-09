setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readr)
library(ggplot2)
library(forecast)
library(tseries)
library(lmtest)


data1 <- read.csv("data.csv")
str(data1)
data2 <- data1[,-1]
str(data2)
na.omit(data2)
summary(data2)
frequency(data2)

#Custom function to implement min max scaling
minMax <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#France
data_FR <- subset(data2, Entity=='France')
data_FR_normalized <- lapply(data_FR[,3:8], minMax)
head(data_FR_normalized)

# 1. Battery 

####Declaring the time series 
ts_FR_battery <- ts(data_FR_normalized$battery_electric_number, start =2001, end=2019, frequency = 1)

####Setting the training set
data_FR_battery <- window(ts_FR_battery, start=2001, end=2016)

par (mfrow = c(1,1))
autoplot (data_FR_battery) + 
  ggtitle ("Battery_electric number in France") + 
  ylab ("battery number") +
  xlab ("Year")

####Stationary test
adf.test(data_FR_battery) #p_value=0.99 the serie is not stationary
Acf(data_FR_battery)
pacf(data_FR_battery)

####Difference to get rid of the trend
ndiffs(data_FR_battery) #2

###auto.arima 
auto.arima(data_FR_battery, trace=TRUE) #The best model is ARIMA(1,2,0) with AICc=-49.17

###Fit the best model 
arima_FR1 <-arima(data_FR_battery, order=c(1,2,0))
summary(arima_FR1) #RMSE 0.03215906
sqrt(arima_FR1$sigma2) #0.03437948

fit1<- fitted(arima_FR1)
plot(data_FR_battery)
lines(fitted(arima_FR1), col=2)

###check residuals 
checkresiduals(arima_FR1)
mean(arima_FR1$residuals)  # this is he MEAN ERROR, the first error type (ME) returned by summary(arima_FR1)

Box.test (arima_FR1$residuals, type = "Ljung-Box", lag = 24)

###forecast
for1<- forecast(arima_FR1)
for1
autoplot(for1)


### Check forecast against actuals
actual_data <- window (ts_FR_battery, start = 2017)
accuracy(for1,actual_data) #RMSE(Test)=0.08881124

#2. Plugin

####Declaring the time series 
ts_FR_plugin <- ts(data_FR_normalized$plugin_hybrid_number, start =2001, end=2019, frequency = 1)

####Setting the training set
data_FR_plugin <- window(ts_FR_plugin, start=2001, end=2016)

par (mfrow = c(1,1))
autoplot (data_FR_plugin) + 
  ggtitle ("Plugin_hybrid number in France") + 
  ylab ("Plugin number") +
  xlab ("Year")

####Stationary test
adf.test(data_FR_plugin) #p_value=0.01823<0.05 the serie is stationary
Acf(data_FR_plugin)
pacf(data_FR_plugin)

###auto.arima 
auto.arima(data_FR_plugin, trace=TRUE) #The best model is ARIMA(0,2,0) with AICc=-52.55

###Fit the best model 
arima_FR2 <-arima(data_FR_plugin, order=c(0,2,0))
summary(arima_FR2) #RMSE 0.03187658
sqrt(arima_FR2$sigma2) #0.03407749

fit1<- fitted(arima_FR2)
plot(data_FR_plugin)
lines(fitted(arima_FR2), col=2)

###check residuals 
checkresiduals(arima_FR2)
mean(arima_FR2$residuals)  # this is he MEAN ERROR, the first error type (ME) returned by summary(arima_FR1)

Box.test (arima_FR1$residuals, type = "Ljung-Box", lag = 24)

###forecast
for2<- forecast(arima_FR2)
for2
autoplot(for2)


### Check forecast against actuals
actual_data2 <- window (ts_FR_plugin, start = 2017)
accuracy(for2,actual_data2) #RMSE(Test)=0.05997754


#3. Full mild

####Declaring the time series 
ts_FR_full <- ts(data_FR_normalized$full_mild_hybrid_number, start =2001, end=2019, frequency = 1)

####Setting the training set
data_FR_full <- window(ts_FR_full, start=2001, end=2016)

par (mfrow = c(1,1))
autoplot (data_FR_full) + 
  ggtitle ("Plugin_hybrid number in France") + 
  ylab ("Plugin number") +
  xlab ("Year")

####Stationary test
adf.test(data_FR_full) #p_value=0.7975<0.05 the serie is not stationary
Acf(data_FR_full)
pacf(data_FR_full)

###auto.arima 
auto.arima(data_FR_full, trace=TRUE) #The best model is ARIMA(0,1,0) with AICc=-35.24

###Fit the best model 
arima_FR3 <-arima(data_FR_full, order=c(0,1,0))
summary(arima_FR3) #RMSE 0.06899851
sqrt(arima_FR3$sigma2) #0.07126136

fit3<- fitted(arima_FR3)
plot(data_FR_full)
lines(fitted(arima_FR3), col=2)

###check residuals 
checkresiduals(arima_FR3)
mean(arima_FR3$residuals)  # this is he MEAN ERROR, the first error type (ME) returned by summary(arima_FR1)

Box.test (arima_FR3$residuals, type = "Ljung-Box", lag = 24)

###forecast
for3<- forecast(arima_FR3)
for3
autoplot(for3)


### Check forecast against actuals
actual_data3 <- window (ts_FR_full, start = 2017)
accuracy(for3,actual_data3) #RMSE(Test)=0.38421000


#4. Petrol

####Declaring the time series 
ts_FR_petrol <- ts(data_FR_normalized$petrol_number, start =2001, end=2019, frequency = 1)

####Setting the training set
data_FR_petrol <- window(ts_FR_petrol, start=2001, end=2016)

par (mfrow = c(1,1))
autoplot (data_FR_petrol) + 
  ggtitle ("Petrol_number in France") + 
  ylab ("Petrol number") +
  xlab ("Year")

####Stationary test
adf.test(data_FR_petrol) #p_value=0.7975<0.05 the serie is not stationary
Acf(data_FR_petrol)
pacf(data_FR_petrol)

###auto.arima 
auto.arima(data_FR_petrol, trace=TRUE) #The best model is ARIMA(1,0,0) with AICc=-35.24

###Fit the best model 
arima_FR4 <-arima(data_FR_petrol, order=c(1,0,0))
summary(arima_FR4) #RMSE 0.1245319
sqrt(arima_FR4$sigma2) #0.1245319

fit4<- fitted(arima_FR4)
plot(data_FR_petrol)
lines(fitted(arima_FR4), col=2)

###check residuals 
checkresiduals(arima_FR4)
mean(arima_FR4$residuals)  # this is he MEAN ERROR, the first error type (ME) returned by summary(arima_FR1)

Box.test (arima_FR4$residuals, type = "Ljung-Box", lag = 24)

###forecast
for4<- forecast(arima_FR4)
for4
autoplot(for4)


### Check forecast against actuals
actual_data4 <- window (ts_FR_petrol, start = 2017)
accuracy(for4,actual_data4) #RMSE(Test)=0.4306222


#5. Diesel

####Declaring the time series 
ts_FR_diesel <- ts(data_FR_normalized$diesel_gas_number, start =2001, end=2019, frequency = 1)

####Setting the training set
data_FR_diesel <- window(ts_FR_diesel, start=2001, end=2016)

par (mfrow = c(1,1))
autoplot (data_FR_diesel) + 
  ggtitle ("diesel_gas_number in France") + 
  ylab ("diesel number") +
  xlab ("Year")

####Stationary test
adf.test(data_FR_diesel) #p_value=0.9277 the serie is not stationary
Acf(data_FR_diesel)
pacf(data_FR_diesel)

###auto.arima 
auto.arima(data_FR_diesel, trace=TRUE) #The best model is ARIMA(0,2,0) with AICc=-35.24

###Fit the best model 
arima_FR5 <-arima(data_FR_diesel, order=c(0,2,0))
summary(arima_FR5) #RMSE 0.09558628 
sqrt(arima_FR5$sigma2) #0.1021859

fit5<- fitted(arima_FR5)
plot(data_FR_diesel)
lines(fitted(arima_FR5), col=2)

###check residuals 
checkresiduals(arima_FR5)
mean(arima_FR5$residuals)  # this is he MEAN ERROR, the first error type (ME) returned by summary(arima_FR1)

Box.test (arima_FR5$residuals, type = "Ljung-Box", lag = 24)

###forecast
for5<- forecast(arima_FR5)
for5
autoplot(for5)

### Check forecast against actuals
actual_data5 <- window (ts_FR_diesel, start = 2017)
accuracy(for5,actual_data5) #RMSE(Test)=0.09455576

