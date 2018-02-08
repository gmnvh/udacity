def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifier
    from sklearn import tree
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)
    
    return clf