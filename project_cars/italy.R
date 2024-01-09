setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)

ds <- read.csv('data.csv')
ds_italy <- subset(ds, Entity=="Italy" )

#omit na value
ds_italy <- na.omit(ds_italy)

#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_italy_norm <- lapply(ds_italy[, 4:9], MinMax)
head(ds_italy_norm)

######electric battery cars###########

#see if there are outliers
par(bty="l")
boxplot(ds_italy$battery_electric_number, main="electric battery")


#apply the differenciation time 
elect_batt_diff1 <- diff(ds_italy_norm$battery_electric_number, differences=1)
plot(ds_italy$Year[1:18], elect_batt_diff1, type="l")

elct.ts1 <- ts(elect_batt_diff1, start=2001, end=2019,  frequency = 1)
ts.plot(elct.ts1, type="l")

## we fit a linear model with the tslm function
fitts1<- tslm(elct.ts1~trend)

plot(elct.ts1, type="l")
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

######plugin cars######

#check outliers
par(bty="l")
boxplot(ds_italy$plugin_hybrid_number, main="plugin hybrid number")

#apply the differenciation time 
plugin_diff1 <- diff(ds_italy_norm$plugin_hybrid_number, differences=1)
plot(ds_italy$Year[1:18], plugin_diff1, type="l")

plugin.ts2 <- ts(plugin_diff1, start=2001, end=2019,  frequency = 1)
ts.plot(plugin.ts2, type="l")

## we fit a linear model with the tslm function
fitts2<- tslm(plugin.ts2~trend)

plot(plugin.ts2, type="l")
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

######## full mild hybrid ########
#check the outliers
par(bty="l")
boxplot(ds_italy$full_mild_hybrid_number, main="fullmild hybrid number")

#apply the differenciation time 
fullmild_diff1 <- diff(ds_italy_norm$full_mild_hybrid, differences=2)
plot(ds_italy$Year[1:17], fullmild_diff1, type="l")

fullmild.ts2 <- ts(fullmild_diff1, start=2001, end=2019,  frequency = 1)
ts.plot(fullmild.ts2, type="l")

## we fit a linear model with the tslm function
fitts2<- tslm(fullmild.ts2~trend)

plot(fullmild.ts2, type="l")
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

############ petrol cars###########
#check outliers
par(bty="l")
boxplot(ds_italy$petrol_number, main="petrol number")

#apply the differenciation time 
petrol_diff1 <- diff(ds_italy_norm$petrol_number, differences=2)
plot(ds_italy$Year[1:17], petrol_diff1, type="l")

petrol.ts2 <- ts(ds_italy_norm$petrol_number, start=2001, end=2019,  frequency = 1)
ts.plot(petrol.ts2, type="l")

## we fit a linear model with the tslm function
fitts2<- tslm(petrol.ts2~trend)

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

###### diesel cars #########
#check outliers
par(bty="l")
boxplot(ds_italy$diesel_gas_number, main="diesel number")

#apply the differenciation time 
diesel_diff1 <- diff(ds_italy_norm$diesel_gas_number, differences=1)
plot(ds_italy$Year[1:18], diesel_diff1, type="l")

diesel.ts2 <- ts(diesel_diff1, start=2001, end=2019,  frequency = 1)
ts.plot(diesel.ts2, type="l")

## we fit a linear model with the tslm function
fitts2<- tslm(diesel.ts2~trend)

plot(diesel.ts2, type="l")
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

#####co2 emissions##########
par(bty="l")
boxplot(ds_italy$co2_per_km, main="co2 emissions")

co2.ts2 <- ts(co2_diff1, start=2001, end=2019,  frequency = 1)
ts.plot(co2.ts2, type="l")

## we fit a linear model with the tslm function
fitts2<- tslm(co2.ts2~trend)

plot(co2.ts2, type="l")
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

