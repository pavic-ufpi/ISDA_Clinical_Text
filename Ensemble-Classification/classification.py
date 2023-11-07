import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize


def classification_kfold(X, y, cls):
    classifier = cls
    cv = KFold(n_splits=10)
    
    actual_classes = np.array([])
    predicted_classes = np.array([])
    probas_classes = list()

    acc_score, kappa_score, f_score, auc_score, prec_score, rec_score = list(), list(), list(), list(), list(), list()

    for train_index, test_index in cv.split(X):
        classifier.fit(X[train_index], y[train_index])
        y_pred = classifier.predict(X[test_index])
        probas = classifier.predict_proba(X[test_index])
        
        actual_classes = np.append(actual_classes, y[test_index])
        predicted_classes = np.append(predicted_classes, y_pred)
        probas_classes.append(probas)

        acc_score.append(accuracy_score(y[test_index], y_pred))
        kappa_score.append(cohen_kappa_score(y[test_index], y_pred))
        f_score.append(f1_score(y[test_index], y_pred, average='macro'))
        prec_score.append(precision_score(y[test_index], y_pred, average='macro'))
        rec_score.append(recall_score(y[test_index], y_pred, average='macro'))     
        auc_score.append(roc_auc_score(y[test_index], probas, multi_class='ovr'))
        
    print('Acurácia média de {:.2f}'.format(sum(acc_score)/len(acc_score)), 'e desvio padrão de {:.2f}'.format(np.std(acc_score)))
    print('Kappa médio de {:.2f}'.format(sum(kappa_score)/len(kappa_score)), 'e desvio padrão de {:.2f}'.format(np.std(kappa_score)))
    print('F1-score médio de {:.2f}'.format(sum(f_score)/len(f_score)), 'e desvio padrão de {:.2f}'.format(np.std(f_score)))
    print('Precisão média de {:.2f}'.format(sum(prec_score)/len(prec_score)), 'e desvio padrão de {:.2f}'.format(np.std(prec_score)))
    print('Recall média de {:.2f}'.format(sum(rec_score)/len(rec_score)), 'e desvio padrão de {:.2f}'.format(np.std(rec_score)))
    print('AUC média de {:.2f}'.format(sum(auc_score)/len(auc_score)), 'e desvio padrão de {:.2f}'.format(np.std(auc_score)))
            
    return classifier, actual_classes, predicted_classes, probas_classes

def metrics(y_test, y_pred, y_pred_proba=None):
    acc_score = accuracy_score(y_test, y_pred)
    kappa_score = cohen_kappa_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred, average='macro')
    prec_score = precision_score(y_test, y_pred, average='macro')
    rec_score = recall_score(y_test, y_pred, average='macro')
    
    print('Acurácia média de {:.2f}'.format(acc_score))
    print('Kappa médio de {:.2f}'.format(kappa_score))
    print('F1-score médio de {:.2f}'.format(f_score))
    print('Precisão média de {:.2f}'.format(prec_score))
    print('Recall média de {:.2f}'.format(rec_score))
    
    if y_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        print('AUC média de {:.2f}'.format(auc_score))