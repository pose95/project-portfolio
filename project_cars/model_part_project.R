setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")
library(readxl)
library(lmtest) 
library(forecast)
library(DIMORA)

ds <- read.csv('data.csv')
ds_france <- subset(ds, Entity=="France" )

#omit na value
ds_france <- na.omit(ds_france)

#see if there are outliers
par(bty="l")
boxplot(ds_france$battery_electric_number, main="electric battery")



#normalize the data with the min-max scaler
MinMax <- function(x){
  (x - min(x)) / (max(x) - min(x))
}
ds_france_norm <- lapply(ds_france[, 4:9], MinMax)
head(ds_france_norm)

#apply the differenciation time 
elect_batt_diff1 <- diff(ds_france_norm$battery_electric_number, differences=2)
plot(ds_france$Year[1:17], elect_batt_diff1, type="l")

########fit a linear regression model 
fit1 <- lm(ds_france$battery_electric_number~ ds_france$Year)
summary(fit1)

##plot of the model
plot(ds_france$Year, ds_france$battery_electric_number, xlab="Time", ylab="electric cars purchasing", type="l")
abline(fit1, col=3)

##check the residuals? are they autocorrelated? Test of DW
dwtest(fit1)

##check the residuals
resfit1<- residuals(fit1)
plot(resfit1,xlab="Year", ylab="residuals", type="l" )

#check the autocorrelation function  on the resdiuals
acf(resfit1)

AIC(fit1)

#####try on min-max scaler data
fit1_norm <- lm(ds_france_norm$battery_electric_number~ ds_france$Year)
summary(fit1)

plot(ds_france$Year, ds_france_norm$battery_electric_number, xlab="Time", ylab="electric cars purchasing", type="l")
abline(fit1_norm, col=3)

dwtest(fit1_norm)

plot(ds_france$Year, ds_france_norm$battery_electric_number, type="l")

#stepwise regression
fit2 <- step(fit1)
summary(fit2)

#predict
p.lm <- predict(fit2)
dev.lm <- sum((p.lm-ds_france$battery_electric_number)^2)
dev.lm

AIC(fit2)

######linear model for time series, with different frecency instead of the above 
elct.ts1 <- ts(ds_france$battery_electric_number, start=2001, end=2019,  frequency = 1)
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


