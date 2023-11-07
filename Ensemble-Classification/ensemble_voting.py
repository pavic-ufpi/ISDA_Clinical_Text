import numpy as np

def hard_voting(predictions, weights=None):
    if weights == None:
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1,arr=predictions,)
    else:
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=weights)), axis=1, arr=predictions,)


def soft_voting(predictions, weights=None):
    if weights == None:
        avg = np.average(predictions, axis=1)
        return np.argmax(avg, axis=1), avg
    else:
        avg = np.average(predictions, axis=1, weights=weights)
        return np.argmax(avg, axis=1), avg
    

def custom_voting(predictions, rf, mlp, svm):
    final_label = list()
    rf=np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1,arr=rf,)
    mlp=np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1,arr=mlp,)
    svm=np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1,arr=svm,)
    
    for i, amostra in enumerate(predictions):  
        label_a = rf[i]
        label_b = mlp[i]
        label_c = svm[i]

        avg = np.average([amostra[0], amostra[1], amostra[2]], axis=0)
        maj = np.argmax(avg)

        if label_a == label_b and label_b == label_c:
            final_label.append(label_a)

        #rf
        elif label_a == label_b and label_b != label_c:
            if maj == label_c:
                final_label.append(label_c)
            else:
                final_label.append(label_a)

        #mlp
        elif label_a == label_c and label_c != label_b:
            if maj == label_b:
                final_label.append(label_b)
            else:
                final_label.append(label_c)

        #svm
        elif label_b == label_c and label_c != label_a:
            if maj == label_a:
                final_label.append(label_a)
            else:
                final_label.append(label_c)

        elif label_a != label_b and label_b != label_c:
            final_label.append(maj)
            
    return np.array(final_label)