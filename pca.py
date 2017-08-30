from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import model_selection
import numpy as np

def pca(df_without_outliers):
    
    df1 = df_without_outliers.copy()    
    
    labels = df1.poi.copy().values
    features = df1.drop("poi", axis = 1).values

    features_names = np.array(df1.drop("poi", axis = 1).columns.tolist())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                    features, labels, test_size = 0.1)    

    selector = SelectPercentile(f_classif, percentile = 15)
    selector.fit(X_train, y_train)
    
    important_features = selector.get_support(indices=False)

    scores = selector.scores_
    
    scores = scores[important_features].tolist() 

    pca_features = features_names[important_features].tolist()

    scores_report =\
	{pca_features[i]:scores[i] for i in range(len(pca_features))}
    
    if 'poi' not in pca_features:
        pca_features.append('poi')

    return pca_features, scores_report
