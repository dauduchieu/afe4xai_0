import numpy as np
import FeatureExtractor as fe
import FeatureSelector as fs
import DecisionTreeModel as dt
def main():
    fe = FeatureExtractor()
    X_train_combined, all_feature_names = fe.transform(train_df)
    X_test_combined, _ = fe.transform(test_df)
    y_train = train_df['condition_label'].values
    y_test = test_df['condition_label'].values

    fs = FeatureSelector(k=20)
    X_train_selected = fs.fit_transform(X_train_combined, y_train)
    X_test_selected = fs.transform(X_test_combined)
    selected_feature_names = fs.get_selected_feature_names(all_feature_names)

    dt = DecisionTreeModel(max_depth=5)
    dt.train(X_train_selected, y_train)
    dt.evaluate(X_train_selected, y_train, "Train")
    dt.evaluate(X_test_selected, y_test, "Test")

    X_train_sorted, sorted_feature_names = dt.sort_features_by_importance(X_train_selected, selected_feature_names)
    X_test_sorted, _ = dt.sort_features_by_importance(X_test_selected, selected_feature_names)

    dt.train_with_sorted_features(X_train_sorted, y_train)
    dt.evaluate(X_train_sorted, y_train, "Sorted Train")
    dt.evaluate(X_test_sorted, y_test, "Sorted Test")

    dt.plot(sorted_feature_names, np.unique(y_train))
    dt.export_rules(sorted_feature_names)

if __name__ == "__main__":
    main()

# This code is a simplified version of the original code and may not run as is.
# It assumes that the necessary data (train_df and test_df) is already loaded and available in the context. 
# The FeatureExtractor, FeatureSelector, and DecisionTreeModel classes are imported from their respective modules.
# The main function orchestrates the feature extraction, selection, training, evaluation, and visualization of the decision tree model.
# The code is designed to be modular, allowing for easy adjustments and improvements in each component.
# The feature extraction process includes the extraction of medical features from text data, while the feature selection process uses SelectKBest to select the top k features based on ANOVA F-value.
# The decision tree model is trained and evaluated on both the original and sorted features, and the results are visualized using matplotlib.
# The decision tree rules are also exported for further analysis.
# Tất cả đoạn tiếng anh ở trên đều là copilot viết, mình không biết nó viết gì, mình chỉ biết là nó viết về decision tree thôi :)) (dòng này cũng thế)
# Các file đều được chạy trên gg colab, mình không chắc là nó có chạy được trên local 1 cách mượt mà hay không.
# Train Accuracy: 0.5676
# Test Accuracy: 0.5478
# Sorted Train Accuracy: 0.5676
# Sorted Test Accuracy: 0.5478
# đây là kết quả của model tốt nhất đối với tập NLLF này sau khi được thử nghiệm, có thể sau khi tối ưu sẽ có model khác phù hợp hơn.
