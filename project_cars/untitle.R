setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)
library(TSstudio)
library(readr)
library(ggplot2)
library(forecast)
library(tseries)
library(lmtest)

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

ts_FR_battery <- ts(ds_france_norm$battery_electric_number, start =2001, end=2019, frequency = 1)
data_FR_battery <- window(ts_FR_battery, start=2001, end=2016)

#diff
elect_batt_diff1 <- diff(data_FR_battery, differences=2)
#plot(ds_france$Year[1:15], elect_batt_diff1, type="l")

adf.test(elect_batt_diff1)

ts.plot(elect_batt_diff1, type="l")

## we fit a linear model with the tslm function
fitts1<- tslm(elect_batt_diff1~trend)

plot(elect_batt_diff1, type="l")
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

### Check forecast against actuals
actual_data2 <- window (ts_FR_battery, start = 2017)
accuracy(forecast1, actual_data2) #RMSE(Test)=0.05997754

AIC(fitts1)
