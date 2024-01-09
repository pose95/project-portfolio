setwd("E:/UniPD Data Science/Business Economic and Financial Data/Project CO2 Vehicle")
######## 5 countries data

co2.car <- read.csv("data5.csv")
str(co2.car)
co2.car <- co2.car[,-1]
summary(co2.car)

# Seperate the data by country

FRANCE <- co2.car[co2.car$Entity == 'France',]
GERMANY <- co2.car[co2.car$Entity == 'Germany',]
ITALY <- co2.car[co2.car$Entity == 'Italy',]
NEDERLAND <- co2.car[co2.car$Entity == 'Netherlands',] 
UK <- co2.car[co2.car$Entity == 'United Kingdom',]

###########---- FRANCE ----############

# train and test set. last 3 obv as test data
FR.train=FRANCE[1:16,]
FR.test=FRANCE[17:19,]

#--------------Battery---------------#

#____Linear Model____

lm.bFR<- lm(battery_electric_number~ co2_per_km, data=FR.train)

summary(lm.bFR)

#Prediction
p.lmbFR <- predict(lm.bFR, newdata=FR.test)
dev.lmbFR <- sum((p.lmbFR-FR.test$battery_electric_number)^2)
dev.lmbFR

AIC(lm.bFR)

#_____GAM_____

library(gam)

# Start with a linear model (df=1)
gam.bFR <- gam(battery_electric_number~co2_per_km, data=FR.train)

# Show the linear effects 
plot(gam.bFR, se=T)

AIC(gam.bFR)

# prediction
p.gambFR <- predict(gam.bFR,newdata=FR.test)     
dev.gambFR <- sum((p.gambFR-FR.test$battery_electric_number)^2)
dev.gambFR

# with spline
gam.bFRs <- gam(battery_electric_number~s(co2_per_km), data=FR.train)
plot(gam.bFRs, se=T)
AIC(gam.bFRs)

# with loess
gam.bFRl <- gam(battery_electric_number~lo(co2_per_km), data=FR.train)
plot(gam.bFRl, se=T)
AIC(gam.bFRl)

# GAM with spline works better. In the following works, only spline will be used


#--------------Plugin---------------#

#____Linear Model____

lm.pFR<- lm(plugin_hybrid_number~ co2_per_km, data=FR.train)

summary(lm.pFR)

#Prediction
p.lmpFR <- predict(lm.pFR, newdata=FR.test)
dev.lmpFR <- sum((p.lmpFR-FR.test$plugin_hybrid_number)^2)
dev.lmpFR

AIC(lm.pFR)

#_____GAM_____

# Start with a linear model (df=1)
gam.pFR <- gam(plugin_hybrid_number~co2_per_km, data=FR.train)
plot(gam.pFR, se=T)

AIC(gam.pFR)

# prediction
p.gampFR <- predict(gam.pFR,newdata=FR.test)     
dev.gampFR <- sum((p.gampFR-FR.test$plugin_hybrid_number)^2)
dev.gampFR

# with spline
gam.pFRs <- gam(plugin_hybrid_number~s(co2_per_km), data=FR.train)
plot(gam.pFRs, se=T)
AIC(gam.pFRs)


#--------------Full Mild---------------#

#____Linear Model____

lm.fFR<- lm(full_mild_hybrid_number~ co2_per_km, data=FR.train)

summary(lm.fFR)

#Prediction
p.lmfFR <- predict(lm.fFR, newdata=FR.test)
dev.lmfFR <- sum((p.lmfFR-FR.test$full_mild_hybrid_number)^2)
dev.lmfFR

AIC(lm.fFR)

#_____GAM_____

# Start with a linear model (df=1)
gam.fFR <- gam(full_mild_hybrid_number~co2_per_km, data=FR.train)
plot(gam.fFR, se=T)

AIC(gam.fFR)

# prediction
p.gamfFR <- predict(gam.fFR,newdata=FR.test)     
dev.gamfFR <- sum((p.gamfFR-FR.test$full_mild_hybrid_number)^2)
dev.gamfFR

# with spline
gam.fFRs <- gam(full_mild_hybrid_number~s(co2_per_km), data=FR.train)
plot(gam.fFRs, se=T)
AIC(gam.fFRs)


#--------------Petrol---------------#

#____Linear Model____

lm.ptFR<- lm(petrol_number~ co2_per_km, data=FR.train)
summary(lm.ptFR)

#Prediction
p.lmptFR <- predict(lm.ptFR, newdata=FR.test)
dev.lmptFR <- sum((p.lmptFR-FR.test$petrol_number)^2)
dev.lmptFR

AIC(lm.ptFR)

#_____GAM_____

# Start with a linear model (df=1)
gam.ptFR <- gam(petrol_number~co2_per_km, data=FR.train)
plot(gam.ptFR, se=T)
AIC(gam.ptFR)

# prediction
p.gamptFR <- predict(gam.ptFR,newdata=FR.test)     
dev.gamptFR <- sum((p.gamptFR-FR.test$petrol_number)^2)
dev.gamptFR

# with spline
gam.ptFRs <- gam(petrol_number~s(co2_per_km), data=FR.train)
plot(gam.ptFRs, se=T)
AIC(gam.ptFRs)


#--------------Diesel---------------#

#____Linear Model____

lm.dFR<- lm(diesel_gas_number~ co2_per_km, data=FR.train)
summary(lm.dFR)

#Prediction
p.lmdFR <- predict(lm.dFR, newdata=FR.test)
dev.lmdFR <- sum((p.lmdFR-FR.test$diesel_gas_number)^2)
dev.lmdFR

AIC(lm.dFR)

#_____GAM_____

# Start with a linear model (df=1)
gam.dFR <- gam(diesel_gas_number~co2_per_km, data=FR.train)
plot(gam.dFR, se=T)
AIC(gam.dFR)

# prediction
p.gamdFR <- predict(gam.dFR,newdata=FR.test)     
dev.gamdFR <- sum((p.gamdFR-FR.test$diesel_gas_number)^2)
dev.gamdFR

# with spline
gam.dFRs <- gam(diesel_gas_number~s(co2_per_km), data=FR.train)
plot(gam.dFRs, se=T)
AIC(gam.dFRs)


########################################
###########---- GERMANY ----############

# train and test set. last 3 obv as test data
DE.train=GERMANY[1:16,]
DE.test=GERMANY[17:19,]

#--------------Battery---------------#

#____Linear Model____

lm.bDE<- lm(battery_electric_number~ co2_per_km, data=DE.train)

summary(lm.bDE)

#Prediction
p.lmbDE <- predict(lm.bDE, newdata=DE.test)
dev.lmbDE <- sum((p.lmbDE-DE.test$battery_electric_number)^2)
dev.lmbDE

AIC(lm.bDE)

#_____GAM_____

library(gam)

# Start with a linear model (df=1)
gam.bDE <- gam(battery_electric_number~co2_per_km, data=DE.train)
plot(gam.bDE, se=T)

AIC(gam.bDE)

# prediction
p.gambDE <- predict(gam.bDE,newdata=DE.test)     
dev.gambDE <- sum((p.gambDE-DE.test$battery_electric_number)^2)
dev.gambDE

# with spline
gam.bDEs <- gam(battery_electric_number~s(co2_per_km), data=DE.train)
plot(gam.bDEs, se=T)
AIC(gam.bDEs)


#--------------Plugin---------------#

#____Linear Model____

lm.pDE<- lm(plugin_hybrid_number~ co2_per_km, data=DE.train)

summary(lm.pDE)

#Prediction
p.lmpDE <- predict(lm.pDE, newdata=DE.test)
dev.lmpDE <- sum((p.lmpDE-DE.test$plugin_hybrid_number)^2)
dev.lmpDE

AIC(lm.pDE)

#_____GAM_____

# Start with a linear model (df=1)
gam.pDE <- gam(plugin_hybrid_number~co2_per_km, data=DE.train)
plot(gam.pDE, se=T)

AIC(gam.pDE)

# prediction
p.gampDE <- predict(gam.pDE,newdata=DE.test)     
dev.gampDE <- sum((p.gampDE-DE.test$plugin_hybrid_number)^2)
dev.gampDE

# with spline
gam.pDEs <- gam(plugin_hybrid_number~s(co2_per_km), data=DE.train)
plot(gam.pDEs, se=T)
AIC(gam.pDEs)


#--------------Full Mild---------------#

#____Linear Model____

lm.fDE<- lm(full_mild_hybrid_number~ co2_per_km, data=DE.train)

summary(lm.fDE)

#Prediction
p.lmfDE <- predict(lm.fDE, newdata=DE.test)
dev.lmfDE <- sum((p.lmfDE-DE.test$full_mild_hybrid_number)^2)
dev.lmfDE

AIC(lm.fDE)

#_____GAM_____

# Start with a linear model (df=1)
gam.fDE <- gam(full_mild_hybrid_number~co2_per_km, data=DE.train)
plot(gam.fDE, se=T)

AIC(gam.fDE)

# prediction
p.gamfDE <- predict(gam.fDE,newdata=DE.test)     
dev.gamfDE <- sum((p.gamfDE-DE.test$full_mild_hybrid_number)^2)
dev.gamfDE

# with spline
gam.fDEs <- gam(full_mild_hybrid_number~s(co2_per_km), data=DE.train)
plot(gam.fDEs, se=T)
AIC(gam.fDEs)


#--------------Petrol---------------#

#____Linear Model____

lm.ptDE<- lm(petrol_number~ co2_per_km, data=DE.train)
summary(lm.ptDE)

#Prediction
p.lmptDE <- predict(lm.ptDE, newdata=DE.test)
dev.lmptDE <- sum((p.lmptDE-DE.test$petrol_number)^2)
dev.lmptDE

AIC(lm.ptDE)

#_____GAM_____

# Start with a linear model (df=1)
gam.ptDE <- gam(petrol_number~co2_per_km, data=DE.train)
plot(gam.ptDE, se=T)
AIC(gam.ptDE)

# prediction
p.gamptDE <- predict(gam.ptDE,newdata=DE.test)     
dev.gamptDE <- sum((p.gamptDE-DE.test$petrol_number)^2)
dev.gamptDE

# with spline
gam.ptDEs <- gam(petrol_number~s(co2_per_km), data=DE.train)
plot(gam.ptDEs, se=T)
AIC(gam.ptDEs)


#--------------Diesel---------------#

#____Linear Model____

lm.dDE<- lm(diesel_gas_number~ co2_per_km, data=DE.train)
summary(lm.dDE)

#Prediction
p.lmdDE <- predict(lm.dDE, newdata=DE.test)
dev.lmdDE <- sum((p.lmdDE-DE.test$diesel_gas_number)^2)
dev.lmdDE

AIC(lm.dDE)

#_____GAM_____

# Start with a linear model (df=1)
gam.dDE <- gam(diesel_gas_number~co2_per_km, data=DE.train)
plot(gam.dDE, se=T)
AIC(gam.dDE)

# prediction
p.gamdDE <- predict(gam.dDE,newdata=DE.test)     
dev.gamdDE <- sum((p.gamdDE-DE.test$diesel_gas_number)^2)
dev.gamdDE

# with spline
gam.dDEs <- gam(diesel_gas_number~s(co2_per_km), data=DE.train)
plot(gam.dDEs, se=T)
AIC(gam.dDEs)


########################################
###########---- ITALY ----############

# train and test set. last 3 obv as test data
IT.train=ITALY[1:16,]
IT.test=ITALY[17:19,]


#--------------Battery---------------#

#____Linear Model____

lm.bIT<- lm(battery_electric_number~ co2_per_km, data=IT.train)

summary(lm.bIT)

#Prediction
p.lmbIT <- predict(lm.bIT, newdata=IT.test)
dev.lmbIT <- sum((p.lmbIT-IT.test$battery_electric_number)^2)
dev.lmbIT

AIC(lm.bIT)

#_____GAM_____

library(gam)

# Start with a linear model (df=1)
gam.bIT <- gam(battery_electric_number~co2_per_km, data=IT.train)
plot(gam.bIT, se=T)

AIC(gam.bIT)

# prediction
p.gambIT <- predict(gam.bIT,newdata=IT.test)     
dev.gambIT <- sum((p.gambIT-IT.test$battery_electric_number)^2)
dev.gambIT

# with spline
gam.bITs <- gam(battery_electric_number~s(co2_per_km), data=IT.train)
plot(gam.bITs, se=T)
AIC(gam.bITs)


#--------------Plugin---------------#

#____Linear Model____

lm.pIT<- lm(plugin_hybrid_number~ co2_per_km, data=IT.train)

summary(lm.pIT)

#Prediction
p.lmpIT <- predict(lm.pIT, newdata=IT.test)
dev.lmpIT <- sum((p.lmpIT-IT.test$plugin_hybrid_number)^2)
dev.lmpIT

AIC(lm.pIT)

#_____GAM_____

# Start with a linear model (df=1)
gam.pIT <- gam(plugin_hybrid_number~co2_per_km, data=IT.train)
plot(gam.pIT, se=T)

AIC(gam.pIT)

# prediction
p.gampIT <- predict(gam.pIT,newdata=IT.test)     
dev.gampIT <- sum((p.gampIT-IT.test$plugin_hybrid_number)^2)
dev.gampIT

# with spline
gam.pITs <- gam(plugin_hybrid_number~s(co2_per_km), data=IT.train)
plot(gam.pITs, se=T)
AIC(gam.pITs)


#--------------Full Mild---------------#

#____Linear Model____

lm.fIT<- lm(full_mild_hybrid_number~ co2_per_km, data=IT.train)

summary(lm.fIT)

#Prediction
p.lmfIT <- predict(lm.fIT, newdata=IT.test)
dev.lmfIT <- sum((p.lmfIT-IT.test$full_mild_hybrid_number)^2)
dev.lmfIT

AIC(lm.fIT)

#_____GAM_____

# Start with a linear model (df=1)
gam.fIT <- gam(full_mild_hybrid_number~co2_per_km, data=IT.train)
plot(gam.fIT, se=T)

AIC(gam.fIT)

# prediction
p.gamfIT <- predict(gam.fIT,newdata=IT.test)     
dev.gamfIT <- sum((p.gamfIT-IT.test$full_mild_hybrid_number)^2)
dev.gamfIT

# with spline
gam.fITs <- gam(full_mild_hybrid_number~s(co2_per_km), data=IT.train)
plot(gam.fITs, se=T)
AIC(gam.fITs)


#--------------Petrol---------------#

#____Linear Model____

lm.ptIT<- lm(petrol_number~ co2_per_km, data=IT.train)
summary(lm.ptIT)

#Prediction
p.lmptIT <- predict(lm.ptIT, newdata=IT.test)
dev.lmptIT <- sum((p.lmptIT-IT.test$petrol_number)^2)
dev.lmptIT

AIC(lm.ptIT)

#_____GAM_____

# Start with a linear model (df=1)
gam.ptIT <- gam(petrol_number~co2_per_km, data=IT.train)
plot(gam.ptIT, se=T)
AIC(gam.ptIT)

# prediction
p.gamptIT <- predict(gam.ptIT,newdata=IT.test)     
dev.gamptIT <- sum((p.gamptIT-IT.test$petrol_number)^2)
dev.gamptIT

# with spline
gam.ptITs <- gam(petrol_number~s(co2_per_km), data=IT.train)
plot(gam.ptITs, se=T)
AIC(gam.ptITs)


#--------------Diesel---------------#

#____Linear Model____

lm.dIT<- lm(diesel_gas_number~ co2_per_km, data=IT.train)
summary(lm.dIT)

#Prediction
p.lmdIT <- predict(lm.dIT, newdata=IT.test)
dev.lmdIT <- sum((p.lmdIT-IT.test$diesel_gas_number)^2)
dev.lmdIT

AIC(lm.dIT)

#_____GAM_____

# Start with a linear moITl (df=1)
gam.dIT <- gam(diesel_gas_number~co2_per_km, data=IT.train)
plot(gam.dIT, se=T)
AIC(gam.dIT)

# prediction
p.gamdIT <- predict(gam.dIT,newdata=IT.test)     
dev.gamdIT <- sum((p.gamdIT-IT.test$diesel_gas_number)^2)
dev.gamdIT

# with spline
gam.dITs <- gam(diesel_gas_number~s(co2_per_km), data=IT.train)
plot(gam.dITs, se=T)
AIC(gam.dITs)


########################################
###########---- NETHERLANDS ----############

# train and test set. last 3 obv as test data
NL.train=NEDERLAND[1:16,]
NL.test=NEDERLAND[17:19,]


#--------------Battery---------------#

#____Linear Model____

lm.bNL<- lm(battery_electric_number~ co2_per_km, data=NL.train)

summary(lm.bNL)

#Prediction
p.lmbNL <- predict(lm.bNL, newdata=NL.test)
dev.lmbNL <- sum((p.lmbNL-NL.test$battery_electric_number)^2)
dev.lmbNL

AIC(lm.bNL)

#_____GAM_____

library(gam)

# Start with a linear model (df=1)
gam.bNL <- gam(battery_electric_number~co2_per_km, data=NL.train)
plot(gam.bNL, se=T)

AIC(gam.bNL)

# prediction
p.gambNL <- predict(gam.bNL,newdata=NL.test)     
dev.gambNL <- sum((p.gambNL-NL.test$battery_electric_number)^2)
dev.gambNL

# with spline
gam.bNLs <- gam(battery_electric_number~s(co2_per_km), data=NL.train)
plot(gam.bNLs, se=T)
AIC(gam.bNLs)


#--------------Plugin---------------#

#____Linear Model____

lm.pNL<- lm(plugin_hybrid_number~ co2_per_km, data=NL.train)

summary(lm.pNL)

#Prediction
p.lmpNL <- predict(lm.pNL, newdata=NL.test)
dev.lmpNL <- sum((p.lmpNL-NL.test$plugin_hybrid_number)^2)
dev.lmpNL

AIC(lm.pNL)

#_____GAM_____

# Start with a linear model (df=1)
gam.pNL <- gam(plugin_hybrid_number~co2_per_km, data=NL.train)
plot(gam.pNL, se=T)

AIC(gam.pNL)

# prediction
p.gampNL <- predict(gam.pNL,newdata=NL.test)     
dev.gampNL <- sum((p.gampNL-NL.test$plugin_hybrid_number)^2)
dev.gampNL

# with spline
gam.pNLs <- gam(plugin_hybrid_number~s(co2_per_km), data=NL.train)
plot(gam.pNLs, se=T)
AIC(gam.pNLs)


#--------------Full Mild---------------#

#____Linear Model____

lm.fNL<- lm(full_mild_hybrid_number~ co2_per_km, data=NL.train)

summary(lm.fNL)

#Prediction
p.lmfNL <- predict(lm.fNL, newdata=NL.test)
dev.lmfNL <- sum((p.lmfNL-NL.test$full_mild_hybrid_number)^2)
dev.lmfNL

AIC(lm.fNL)

#_____GAM_____

# Start with a linear model (df=1)
gam.fNL <- gam(full_mild_hybrid_number~co2_per_km, data=NL.train)
plot(gam.fNL, se=T)

AIC(gam.fNL)

# prediction
p.gamfNL <- predict(gam.fNL,newdata=NL.test)     
dev.gamfNL <- sum((p.gamfNL-NL.test$full_mild_hybrid_number)^2)
dev.gamfNL

# with spline
gam.fNLs <- gam(full_mild_hybrid_number~s(co2_per_km), data=NL.train)
plot(gam.fNLs, se=T)
AIC(gam.fNLs)


#--------------Petrol---------------#

#____Linear Model____

lm.ptNL<- lm(petrol_number~ co2_per_km, data=NL.train)
summary(lm.ptNL)

#Prediction
p.lmptNL <- predict(lm.ptNL, newdata=NL.test)
dev.lmptNL <- sum((p.lmptNL-NL.test$petrol_number)^2)
dev.lmptNL

AIC(lm.ptNL)

#_____GAM_____

# Start with a linear model (df=1)
gam.ptNL <- gam(petrol_number~co2_per_km, data=NL.train)
plot(gam.ptNL, se=T)
AIC(gam.ptNL)

# prediction
p.gamptNL <- predict(gam.ptNL,newdata=NL.test)     
dev.gamptNL <- sum((p.gamptNL-NL.test$petrol_number)^2)
dev.gamptNL

# with spline
gam.ptNLs <- gam(petrol_number~s(co2_per_km), data=NL.train)
plot(gam.ptNLs, se=T)
AIC(gam.ptNLs)


#--------------Diesel---------------#

#____Linear Model____

lm.dNL<- lm(diesel_gas_number~ co2_per_km, data=NL.train)
summary(lm.dNL)

#Prediction
p.lmdNL <- predict(lm.dNL, newdata=NL.test)
dev.lmdNL <- sum((p.lmdNL-NL.test$diesel_gas_number)^2)
dev.lmdNL

AIC(lm.dNL)

#_____GAM_____

# Start with a linear model (df=1)
gam.dNL <- gam(diesel_gas_number~co2_per_km, data=NL.train)
plot(gam.dNL, se=T)
AIC(gam.dNL)

# prediction
p.gamdNL <- predict(gam.dNL,newdata=NL.test)     
dev.gamdNL <- sum((p.gamdNL-NL.test$diesel_gas_number)^2)
dev.gamdNL

# with spline
gam.dNLs <- gam(diesel_gas_number~s(co2_per_km), data=NL.train)
plot(gam.dNLs, se=T)
AIC(gam.dNLs)


########################################
###########---- UK ----############

# train and test set. last 3 obv as test data
UK.train=UK[1:16,]
UK.test=UK[17:19,]

#--------------Battery---------------#

#____Linear Model____

lm.bUK<- lm(battery_electric_number~ co2_per_km, data=UK.train)

summary(lm.bUK)

#Prediction
p.lmbUK <- predict(lm.bUK, newdata=UK.test)
dev.lmbUK <- sum((p.lmbUK-UK.test$battery_electric_number)^2)
dev.lmbUK

AIC(lm.bUK)

#_____GAM_____

# Start with a linear model (df=1)
gam.bUK <- gam(battery_electric_number~co2_per_km, data=UK.train)
plot(gam.bUK, se=T)

AIC(gam.bUK)

# prediction
p.gambUK <- predict(gam.bUK,newdata=UK.test)     
dev.gambUK <- sum((p.gambUK-UK.test$battery_electric_number)^2)
dev.gambUK

# with spline
gam.bUKs <- gam(battery_electric_number~s(co2_per_km), data=UK.train)
plot(gam.bUKs, se=T)
AIC(gam.bUKs)


#--------------Plugin---------------#

#____Linear Model____

lm.pUK<- lm(plugin_hybrid_number~ co2_per_km, data=UK.train)

summary(lm.pUK)

#Prediction
p.lmpUK <- predict(lm.pUK, newdata=UK.test)
dev.lmpUK <- sum((p.lmpUK-UK.test$plugin_hybrid_number)^2)
dev.lmpUK

AIC(lm.pUK)

#_____GAM_____

# Start with a linear model (df=1)
gam.pUK <- gam(plugin_hybrid_number~co2_per_km, data=UK.train)
plot(gam.pUK, se=T)

AIC(gam.pUK)

# prediction
p.gampUK <- predict(gam.pUK,newdata=UK.test)     
dev.gampUK <- sum((p.gampUK-UK.test$plugin_hybrid_number)^2)
dev.gampUK

# with spline
gam.pUKs <- gam(plugin_hybrid_number~s(co2_per_km), data=UK.train)
plot(gam.pUKs, se=T)
AIC(gam.pUKs)


#--------------Full Mild---------------#

#____Linear Model____

lm.fUK<- lm(full_mild_hybrid_number~ co2_per_km, data=UK.train)

summary(lm.fUK)

#Prediction
p.lmfUK <- predict(lm.fUK, newdata=UK.test)
dev.lmfUK <- sum((p.lmfUK-UK.test$full_mild_hybrid_number)^2)
dev.lmfUK

AIC(lm.fUK)

#_____GAM_____

# Start with a linear model (df=1)
gam.fUK <- gam(full_mild_hybrid_number~co2_per_km, data=UK.train)
plot(gam.fUK, se=T)

AIC(gam.fUK)

# prediction
p.gamfUK <- predict(gam.fUK,newdata=UK.test)     
dev.gamfUK <- sum((p.gamfUK-UK.test$full_mild_hybrid_number)^2)
dev.gamfUK

# with spline
gam.fUKs <- gam(full_mild_hybrid_number~s(co2_per_km), data=UK.train)
plot(gam.fUKs, se=T)
AIC(gam.fUKs)


#--------------Petrol---------------#

#____Linear Model____

lm.ptUK<- lm(petrol_number~ co2_per_km, data=UK.train)
summary(lm.ptUK)

#Prediction
p.lmptUK <- predict(lm.ptUK, newdata=UK.test)
dev.lmptUK <- sum((p.lmptUK-UK.test$petrol_number)^2)
dev.lmptUK

AIC(lm.ptUK)

#_____GAM_____

# Start with a linear model (df=1)
gam.ptUK <- gam(petrol_number~co2_per_km, data=UK.train)
plot(gam.ptUK, se=T)
AIC(gam.ptUK)

# prediction
p.gamptUK <- predict(gam.ptUK,newdata=UK.test)     
dev.gamptUK <- sum((p.gamptUK-UK.test$petrol_number)^2)
dev.gamptUK

# with spline
gam.ptUKs <- gam(petrol_number~s(co2_per_km), data=UK.train)
plot(gam.ptUKs, se=T)
AIC(gam.ptUKs)


#--------------Diesel---------------#

#____Linear Model____

lm.dUK<- lm(diesel_gas_number~ co2_per_km, data=UK.train)
summary(lm.dUK)

#Prediction
p.lmdUK <- predict(lm.dUK, newdata=UK.test)
dev.lmdUK <- sum((p.lmdUK-UK.test$diesel_gas_number)^2)
dev.lmdUK

AIC(lm.dUK)

#_____GAM_____

# Start with a linear model (df=1)
gam.dUK <- gam(diesel_gas_number~co2_per_km, data=UK.train)
plot(gam.dUK, se=T)
AIC(gam.dUK)

# prediction
p.gamdUK <- predict(gam.dUK,newdata=UK.test)     
dev.gamdUK <- sum((p.gamdUK-UK.test$diesel_gas_number)^2)
dev.gamdUK

# with spline
gam.dUKs <- gam(diesel_gas_number~s(co2_per_km), data=UK.train)
plot(gam.dUKs, se=T)
AIC(gam.dUKs)




