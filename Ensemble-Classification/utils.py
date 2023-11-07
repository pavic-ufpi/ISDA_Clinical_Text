import numpy as np
import seaborn as sns
from itertools import cycle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list, combination):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    labels_names = ['Prescriptions', 'Clinical\nnotes', 'Medical\nexamination']
    
    hm = sns.heatmap(matrix, cmap="Blues", fmt="g", annot=True)
    hm.set_xticklabels(labels=labels_names, rotation=0, fontsize=11)
    hm.set_yticklabels(labels=labels_names, rotation=0, fontsize=11)
    
    plt.xlabel('Predicted', fontsize=11); 
    plt.ylabel('Actual', fontsize=11); 
    plt.title('Confusion Matrix', fontsize=13)
    
    filename = 'C:\\Users\\orran\\OneDrive\\Documentos\\GitHub\\Research-Prescriptions\\3k\\Experimento\\Ensemble-Classification\\figures\\confusion_matrix_%s.png' %combination
    plt.tight_layout()
    plt.savefig(filename)
    
def plot_roc_curve(actual_classes : np.array, predicted_classes : np.array, combination):

    y_test = label_binarize(actual_classes, classes=[0, 1, 2])
    y_pred = label_binarize(predicted_classes, classes=[0, 1, 2])
    
    n_classes = 3 

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Binary plot
    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    filename = 'C:\\Users\\orran\\OneDrive\\Documentos\\GitHub\\Research-Prescriptions\\3k\\Experimento\\Ensemble-Classification\\figures\\roc_curve_binary_%s.png' %combination
    plt.savefig(filename)
    plt.show()
    
    # Multiclass plot
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    filename = 'C:\\Users\\orran\\OneDrive\\Documentos\\GitHub\\Research-Prescriptions\\3k\\Experimento\\Ensemble-Classification\\figures\\roc_curve_multiclass_%s.png' %combination
    plt.savefig(filename)
    plt.show()

def plot_proj(embedding, lbs, combination):
    n = len(embedding)
    counter = Counter(lbs)
    nome = ''
    cor = ''
    for i in range(len(np.unique(lbs))):
        if i == 0:
            nome = 'Prescriptions'
        elif i == 1:
            nome = 'Clinical notes'
        else:
            nome = 'Med. examinations'  #, label='{}: {:.2f}%'.format(nome, counter[i] / n * 100)
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', label='{}: {:.2f}%'.format(nome, counter[i] / n * 100))
    plt.legend(loc = 'best', fontsize=8)
    plt.xlabel('First component', fontsize=11)
    plt.ylabel('Second component', fontsize=11)
    plt.grid(color ='grey', linestyle='-',linewidth = 0.25)
    filename = 'C:\\Users\\orran\\OneDrive\\Documentos\\GitHub\\Research-Prescriptions\\Experimento\\Ensemble-Classification\\figures\\pca_%s.png' %combination
    plt.savefig(filename, dpi=800)