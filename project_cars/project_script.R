setwd("C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars")

library(ggplot2)
library(ggplot)
library(dplyr)

#import the two datasets
ds_auto <- read.csv("new-vehicles-type-area.csv")
ds_carbon_emission <- read.csv("carbon-new-passenger-vehicles.csv")
ds_def <- read.csv("ds_def.csv")
#clean the two dataset (remove code of the country)
ds_auto <- ds_auto[,-2]
ds_carbon_emission <- ds_carbon_emission[,-2]

#merge the two datasets in one 

ds_def <- merge(ds_auto, ds_carbon_emission, by=c("Year", "Entity"), all.x=TRUE)

write.csv(ds_def, "C:\\Users\\matteo posenato\\Documents\\Business economical and finacial data\\project_cars\\ds_definitve.csv")


#exploratory data analysis
#boxplot of countries
plot(ds_def$Entity, ds_def$Year)
#heatmap
groupSummary <- ddply(incidents, c( "co2_per_km", "Year"), summarise,
                      N    = length(ymd)
)

#overall summary
ggplot(groupSummary, aes( hour,Event.Clearance.Group)) + geom_tile(aes(fill = N),colour = "white") +
  scale_fill_gradient(low = col1, high = col2) +  
  guides(fill=guide_legend(title="Total Incidents")) +
  labs(title = "Histogram of Seattle Incidents by Event and Hour",
       x = "Hour", y = "Event") +
  theme_bw() + theme_minimal() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
#plot(Austria_emission$co2_per_km~Austria_emission$Year, type ="l")

#ggplot() +
  #geom_line(aes(x=Austria_emission$Year, y=Austria_emission$co2_per_km), col="red") +
  #geom_line(aes(x=Austria_emission$Year, y=Austria_car$petrol_number), col="blue")
  #scale_y_continuous(
    #name="num_car",
    #sec.axis= sec_axis(~./10000, name="num_car"))

#plot the diffentents type of cars to see if the data start from the same years
plot(ds_def$Year, ds_def$battery_electric_number)
