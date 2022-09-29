# EECS738-Proj_01

In this project, I developed the Gaussian Mixture Model from scratch and used the model to cluster data into different groups based on the similarities of two features from the data. We can use the GMM model to predict the wine quality for wine quality data, and predict student's final grade from student performance data by clustering. 

# Project Instruction

Project 1 - Probably Interesting Data  
Distribution estimation  
1. Set up a new git repository in your GitHub account  
2. Pick two datasets from https://www.kaggle.com/uciml/datasets  
3. Choose a programming language (Python, C/C++, Java)  
4. Formulate ideas on how machine learning can be used to model distributions within the dataset  
5. Build a heuristic and/or algorithm to model the data using mixture models of probability distributions programmatically  
6. Document your process and results  
7. Commit your source code, documentation and other supporting files to the git repository in GitHub  

# Dataset

Wine Quality: https://archive.ics.uci.edu/ml/datasets/Wine+Quality  

Student Performance: https://archive.ics.uci.edu/ml/datasets/Student+Performance  

# Results

As result, my model connot converge for multiple iterations, the value of mu and sigma will become super large when more iterations are run. But when I only run with 1 iteration, the data is clustered perfectly into different groups. I tested the data with both 2 clusters and 3 clusters.

# References

https://brilliant.org/wiki/gaussian-mixture-model/  
https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95  
https://stephens999.github.io/fiveMinuteStats/intro_to_em.html  
