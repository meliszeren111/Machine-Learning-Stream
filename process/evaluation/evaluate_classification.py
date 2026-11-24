from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, \
    precision_score, recall_score, average_precision_score, roc_curve, auc


def evaluate_classification(y_test, y_pred, y_prob=None):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    roc_auc = None
    average_precision = None
    roc_auc_auc = None

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr" if len(set(y_test)) > 2 else "raise")
        average_precision = average_precision_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc_auc = auc(fpr, tpr)

    return {

            "accuracy": accuracy,  # Ratio of correctly predicted observations to total observations
            "confusion_matrix": conf_matrix,  # Table showing actual versus predicted classifications to evaluate errors
            "classification_report": class_report,  # Text summary of precision, recall, and F1-score for each class
            "f1_score": f1,  # Harmonic mean of precision and recall balancing both metrics
            "precision": precision,  # Ratio of correctly predicted positive observations to total predicted positives
            "recall": recall,  # Ratio of correctly predicted positive observations to all actual positives
            "roc_auc_score": roc_auc, # Area under the Receiver Operating Characteristic curve measuring model separability
            "average_precision": average_precision,# Summarizes the precision-recall curve as the weighted mean of precisions
            "auc_score": roc_auc_auc  # Area under the ROC curve calculated from false and true positive rates
         }













