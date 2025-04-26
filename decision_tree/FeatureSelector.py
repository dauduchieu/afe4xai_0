from sklearn.feature_selection import SelectKBest, f_classif

class FeatureSelector:
    def __init__(self, k=20):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)

    def fit_transform(self, X, y):
        print("Selecting best features...")
        X_selected = self.selector.fit_transform(X, y)
        return X_selected

    def transform(self, X):
        return self.selector.transform(X)

    def get_selected_feature_names(self, all_feature_names):
        indices = self.selector.get_support(indices=True)
        return [all_feature_names[i] for i in indices]