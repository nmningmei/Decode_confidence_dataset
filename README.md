# Decoding analyses of the confidence dataset via linear support vector machine, random forest classifier and recurrent neural network models.

# System Information
- Platform:      Linux-3.10.0-514.el7.x86_64-x86_64-with-centos-7.3.1611-Core
- Python:        3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42)  [GCC 7.2.0]
- CPU:           x86_64: 16 cores
- numpy:         1.16.4 {blas=mkl_rt, lapack=mkl_rt}
- scipy:         1.3.1
- matplotlib:    3.1.3 {backend=agg}
- seaborn:       0.11.1
- sklearn:       0.23.2
- pandas:        1.0.1
- tensorflow:    2.0.0
- pytorch:       1.7.1
- R:             4.0.3 # for 3-way repeated measure ANOVAs

# Linear SVM
<img src="https://image.slideserve.com/867897/linear-support-vector-machine-svm-l.jpg" width="50%" />
source: https://image.slideserve.com/867897/linear-support-vector-machine-svm-l.jpg

# Random forest classifier
<img src="https://cdn-images-1.medium.com/max/1600/1*i0o8mjFfCn-uD79-F1Cqkw.png">
source: https://cdn-images-1.medium.com/max/1600/1*i0o8mjFfCn-uD79-F1Cqkw.png

# RNN model - as an alternative model, but we do not perform model selection. An RNN model contains such prior: there exists temporal relationship between the features from consective time points and adding these relationships to the model would benefit the decoding. 
<p float="left">
  <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/RNN%20model%20confidence%20database.jpg" width="40%" /> <img src="https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/external-content.duckduckgo.com.jpg" width="40%" />
</p>


# Results - confidence
## Decoding scores (within domain)
Using 7 trials back             |  Split to past and recent
:-------------------------:|:-------------------------:
![t7s](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO/scores.jpg)  |  ![prs](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO/scores_split.jpg)


## Feature attribution (within domain)
Using 7 trials back - SVM             |  Using 7 trials back - RF
:-------------------------:|:-------------------------:
![t7w](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO/features.jpg)  |  ![t7f](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/LOO/feature_importance.jpg)


## Decoding scores (cross domain)
Using 7 trials back             |  Split to past and recent
:-------------------------:|:-------------------------:
![t7sc](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/cross_domain/scores.jpg)  |  ![prsc](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/cross_domain/scores_split.jpg)


## Feature attribution (cross domain)
Using 7 trials back - SVM             |  Using 7 trials back - RF
:-------------------------:|:-------------------------:
![t7wc](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/cross_domain/features.jpg)  |  ![t7fc](https://github.com/nmningmei/decoding_confidence_dataset/blob/main/figures/confidence/cross_domain/feature_importance.jpg)



