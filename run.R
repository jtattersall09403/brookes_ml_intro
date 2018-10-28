# This file should be used to run all of the code required to populate the Outputs folder 

# Script to download data and run dimension reduction
source('./R/01 Data processing and DR.R')

# Run KNN experiment


# Renders report markdown in Outputs folder 

rmarkdown::render('report.Rmd', output_format = 'html_document', output_file='report.html', output_dir='Outputs')
