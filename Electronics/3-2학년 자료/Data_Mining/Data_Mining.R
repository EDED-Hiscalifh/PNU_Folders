# Midterm Project of Data Mining 

# Preparation
#install.packages('dslabs')
#install.packages('ggplot2')
#install.packages('tidyr')

library(dslabs)
library(ggplot2)
library(tidyr)

path1 <- "Data_Mining/Datasets/breast-cancer.txt"
path2 <- "Data_Mining/Datasets/Teaching Assistant Evaluation.txt"
path3 <- "Data_Mining/Datasets/Tic-Tac-Toe Endgame.txt"

bc_dataset <- read.table(path1, sep = ',')
tae_dataset <- read.table(path2, sep = ',')
ttt_dataset <- read.table(path3, sep = ',')

# 1. breast-cancer
str(bc_dataset)

# 2. Teaching Assistant Evaluation
str(tae_dataset)

# 3. Tic-Tac-Toe Endgame
str(ttt_dataset)