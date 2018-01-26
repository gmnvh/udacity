# Lesson 3 - SVM

---
 - Support Vector Machines
 - Margin = Maximizes distance to nearest point
 - [x] Download quiz 12 and run in the pc
    - Accuracy for quiz 12 - SVM for terrain : 0.92
 - Effect of gamma and C for rbf SVMs (Quiz 22)
    - http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    - Besides the fact that the answer for quiz 22 is that a large C should get more points correct. 
    The accuracy for a large C is only better if gamma is low. C=1 ended up having a better performance if gamma is big.
    If gamma is 1 and C is 1000, the accuracy is 0.9399. The best classifier tried so far.
 - gamma - defines how far the influence of a single training example reaches
    - low values = far
    - high values = close
 - SVM works well on clear domains but does not perform so well with big data sets and noise data
 
 ---

## Mini Project
  ### Goal

  In this mini-project, we’ll tackle the exact same email author ID problem as the Naive Bayes mini-project, 
  but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. 
  This project also gives us a chance to play around with parameters a lot more than Naive Bayes did, so we will do 
  that too.
  
  ### Steps
    - On svm/svm_author_id.py implement a SVM using linear kernel. What is the accuracy of the classifier ? (L3Q28)
        - Linear kernel - accuracy  0.984072810011
    - Are the SVM training and predicting times faster or slower than Naive Bayes ? (L3Q29)
        - Fit and prediction time: 162.57 s
    - Re-do with just 1% of the data. What the accuracy now ? (L3Q30)
        - Linear kernel with 1% of the data - accuracy  0.884527872582
        - Fit and prediction time: 0.954 s
    - Keep the 1% data and change kernel to "rbf". What is the accuracy ? (L3Q32)
        - Rbf kernel with 1% of the data - accuracy  0.616040955631
        - Fit and prediction time: 1.04 s
    - Keep the training set size and rbf kernel from the last quiz, but try several values of C
        (say, 10.0, 100., 1000., and 10000.). Which one gives the best accuracy? (L3Q33)
         - C = 10000 has the best accuracy of 0.8924
    - Using C=10000, train using all the database (L3Q35)
        - Rbf kernel and C=10000 - accuracy  0.990898748578
        - Fit and prediction time: 104.891 s
    - Using the reduced dataset, show positions 10, 26 and 50 (L3Q36)
        - 1 0 1
    - There are over 1700 test events--how many are predicted to be in the “Chris” (1) class?
    (Use the RBF kernel, C=10000., and the full training set.) (L3Q37)
        - Prediction length:  1758
        - Predictions for Chris(1) 877

  ### To-Do
  - [x] Update wiki page
  - [x] Run and publish the results using Jupyter
