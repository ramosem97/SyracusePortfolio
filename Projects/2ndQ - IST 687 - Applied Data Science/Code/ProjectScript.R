## Project Starts
library(data.table)
library(dplyr)
library(ggplot2)
setwd("/Users/ramosem/Documents/SyracuseUniversity/2nd_Quarter/IST687/Project")

df <- read.csv("ColombiaForeignAssistanceRAW.csv", header = TRUE)

head(df)

colnames(df) <- c("Agency", "ReceivingLocation", "TransactionValue", "FiscalYear", "Quarter", "OECD", "AssistanceSectorCode")

## 1)
## What was the variance of U.S foreign assistance between fiscal years 2014 to 2019 in Colombia?

dfAgg <- aggregate(.~df$FiscalYear+df$Quarter,  df %>% select(TransactionValue), sum)

colnames(dfAgg) <- c("FiscalYear", "Quarter", "Value")

dfAgg <- dfAgg[dfAgg$FiscalYear >= 2014, ]

dfAgg$FiscalYear <- as.character(dfAgg$FiscalYear)
dfAgg$Quarter <- as.character(dfAgg$Quarter)

dfAgg$Time <- seq(length(dfAgg$FiscalYear))

model <- lm(Value ~ Time, dfAgg)

summary(model)

ggplot(dfAgg, aes(x = Time, y = Value)) +
  geom_smooth()     


dfAggYear  <- aggregate(.~df$FiscalYear,  df %>% select(TransactionValue), sum)
colnames(dfAggYear) <- c("FiscalYear", "Value")
model2 <- lm(Value ~ FiscalYear, dfAggYear)

summary(model2)

ggplot(dfAggYear, aes(x = FiscalYear, y = Value)) +
  geom_smooth()

ggplot(dfAgg, aes(x = FiscalYear, y = Value, fill = Quarter)) +
  geom_col(position = "identity")     

## 2)
## What was the variance of U.S foreign assistance between fiscal years 2014 to 2019 in Colombia?
df14 <- df[df$FiscalYear >= 2014,]

dfAggTotal <- aggregate(.~df14$OECD, df14 %>% select(TransactionValue), sum)

colnames(dfAggTotal) <- c("OECD", "Value")

View(dfAggTotal[order(-dfAggTotal$Value), ])

top5OECD = dfAggTotal[order(-dfAggTotal$Value), ]$OECD[c(1:5)]


for (oecd in top5OECD) {
  print(oecd)
  df14 <- df[(df$FiscalYear >= 2014) & (df$OECD == oecd),]
  
  dfAggAgency <- aggregate(.~df14$FiscalYear+df14$Quarter+df14$Agency, df14 %>% select(TransactionValue), sum)
  colnames(dfAggAgency) <- c("FiscalYear", "Quarter", "Agency", "Value")
  dfAggAgency$Percentage <- 100*dfAggAgency$Value / sum(dfAggAgency$Value)
  
  dfAgg <- aggregate(.~df14$FiscalYear+df14$Quarter, df14 %>% select(TransactionValue), sum)
  colnames(dfAgg) <- c("FiscalYear", "Quarter", "Value")
  
  dfAgg$FiscalYear <- as.character(dfAgg$FiscalYear)
  dfAgg$Quarter <- as.character(dfAgg$Quarter)
  
  pltVar <- ggplot(dfAgg, aes(x = FiscalYear, y = Value, fill = Quarter)) +
    geom_col(position = "identity")  +
    ggtitle(oecd)
  print(oecd)
  print(head(dfAggAgency[order(-dfAggAgency$Percentage), ]))
  bp<- ggplot(dfAggAgency, aes(x="", y=Percentage, fill=Agency))+
    geom_bar(width = 1, stat = "identity")
  
  pie <- bp + coord_polar("y", start=0)  +
    ggtitle(oecd)
  
  print(pltVar)
  print(pie)
  
}


## 3)
## What was the variance of U.S foreign assistance between fiscal years 2014 to 2019 in Colombia?
df14 <- df[df$FiscalYear >= 2014,]

dfAggTotal <- aggregate(.~df14$Agency, df14 %>% select(TransactionValue), sum)

colnames(dfAggTotal) <- c("Agency", "Value")

View(dfAggTotal[order(-dfAggTotal$Value), ])

top5Agency = dfAggTotal[order(-dfAggTotal$Value), ]$Agency[c(1:5)]


for (agency in top5Agency) {
  print(agency)
  df14 <- df[(df$FiscalYear >= 2014) & (df$Agency == agency),]
  
  dfAggOEDC <- aggregate(.~df14$FiscalYear+df14$Quarter+df14$OECD, df14 %>% select(TransactionValue), sum)
  colnames(dfAggOEDC) <- c("FiscalYear", "Quarter", "OECD", "Value")
  dfAggOEDC$Percentage <- 100*dfAggOEDC$Value / sum(dfAggOEDC$Value)
  
  dfAgg <- aggregate(.~df14$FiscalYear+df14$Quarter, df14 %>% select(TransactionValue), sum)
  colnames(dfAgg) <- c("FiscalYear", "Quarter", "Value")
  
  dfAgg$FiscalYear <- as.character(dfAgg$FiscalYear)
  dfAgg$Quarter <- as.character(dfAgg$Quarter)
  
  
  pltVar <- ggplot(dfAgg, aes(x = FiscalYear, y = Value, fill = Quarter)) +
    geom_col(position = "identity")  +
    ggtitle(agency)
  
  
  dfAggOEDC2 <- aggregate(.~df14$OECD, df14 %>% select(TransactionValue), sum)
  colnames(dfAggOEDC2) <- c("OECD", "Value")
  dfAggOEDC2$Percentage <- 100*dfAggOEDC2$Value / sum(dfAggOEDC2$Value)
  
  bp<- ggplot(dfAggOEDC2, aes(x="", y=Percentage, fill=OECD))+
    geom_bar(width = 1, stat = "identity")
  
  pie <- bp + coord_polar("y", start=0)  +
    ggtitle(agency)
  print(agency)
  print(head(dfAggOEDC2[order(-dfAggOEDC2$Percentage), ]))
  
  print(pltVar)
  print(pie)
  
} 

for (agency in top5Agency) {
  print(agency)
  
  df14 <- df[(df$FiscalYear >= 2014) & (df$Agency == agency),]
  
  dfAgg <- aggregate(.~df14$FiscalYear+df14$Quarter+df14$OECD, df14 %>% select(TransactionValue), sum)
  colnames(dfAgg) <- c("FiscalYear", "Quarter", "OECD", "Value")
  
  dfAgg$FiscalQuarter <- paste(as.character(dfAgg$FiscalYear), as.character(dfAgg$Quarter))
  
  pltVar <- ggplot(dfAgg, aes(x = FiscalQuarter, y = Value, fill = OECD)) +
    geom_col(position = "identity")  +
    ggtitle(paste(agency, " over time for each year"))
  View(dfAgg)
  print(pltVar)
  print(head())
}




