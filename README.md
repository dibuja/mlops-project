Classification of Yelp business reviews based on sentiment analysis.
==============================

This repository contains a project conducted within the scope of the DTU's course "Machine Learning Operations". The structure of the project follows the [cookie cutter structure](http://drivendata.github.io/cookiecutter-data-science/).

The topic of the project consists of classifying reviews of businesses and try to predict the number of stars a review has based on the sentiment analysis of the comments left by its author. For that purpose, the BERT language model taken from the Transformers repository is used.

The following MLOps practices have been – or are being – implemented:

- Using the cookie cutter structure
- Pre-commits with code formatting and checking
- Separation of the algorithms in make_dataset, model file and training scripts
- Requirements.txt file
- Dedicated Conda environment for the project
- Compliance with pep8 coding conventions
- Usage of Google Cloud Platform for data storage, model deployment
- (...)