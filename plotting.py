import seaborn
import sklearn
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def plotting(x, y, x_label, y_label, x_lim = [0, 1], y_lim = [0, 1],  name = 'Example', filename = None, baseline_val = None, perfect_val = None, legend_loc = 'lower right'):
    """"Help function for plotting ROC and PR curves. 

    Parameters:
        x (array like object): Values to plot on horizontal axis.
        y (array like object): Values to plot on vertical axis. 
        x_label (str): Label on horizontal axis. 
        y_label (str): Label on vertical axis. 
        x_lim (list): Range of horizontal axis.
        y_lim (list): Range of vertical axis.
        name (str): Entry to ledgend of curve. 
        filename (str): Name of file to sace figure to. None if do not want to save fig. 
        baseline_val (array like object): Baseline to plot. None if do not print a basline. 
        perfect_val (list): List with location to plot perfect x and y values. First element x, second y. None if do not print a perfect value. 
        legend_loc (str): Location where to print ledgend. 
    """
    plt.figure()
    # plot "optimal" classifer
    if perfect_val is not None:
        plt.axvline(x=perfect_val[0], label='Optimal', ymin = 0, ymax = perfect_val[-1], color = 'darkgreen')
        plt.axhline(y=perfect_val[-1], xmin = 0, xmax = perfect_val[-1], color = 'darkgreen')
        
    plt.plot(x, y, linewidth=2, label = name, color = 'midnightblue')
    if baseline_val is not None:
        plt.plot(x, baseline_val, linewidth=2, label = 'Baseline', color = 'red')

    label_size = 16
    plt.xlabel(x_label, size=label_size)
    plt.ylabel(y_label, size=label_size)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc=legend_loc, fontsize= 14)
    ax = plt.gca()
    if x_lim == y_lim: ax.set_aspect('equal')
    # save PNG file
    if filename is not None: plt.savefig('./Figures/'+filename+'.png', bbox_inches='tight')
    else: plt.show()


def plot_roc(name, labels, predictions):
    """"Plots ROC curve form labels and predictions. 

    Parameters:
        labels (array like object): Correct labels
        predictions (array like object): Predicted values. 
    Returns:
        list: The false positive rate.
        list: The true positive rate.
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
    baseline = fpr
    plotting(fpr, tpr, 'False positive rate', 'True positive rate', baseline_val=baseline, name=name)
    return fpr.tolist(), tpr.tolist()
             
def plot_prc(name, labels, predictions):
    """"Plots precision-recall curve form labels and predictions. 

    Parameters:
        name (str): 
        labels (ndarray): Correct labels
        predictions (ndarray): Predicted values. 
    Returns:
        list: The precision at all threholds
        list: The recall at all thresholds
    """
    p, r, _ = sklearn.metrics.precision_recall_curve(labels, predictions) 
    baseline = np.ones(r.shape) * (np.mean(labels)) # (number of true labels in dataset)/(number samples in dataset)
    plotting(r, p, 'Recall', 'Precision', baseline_val=baseline, name=name, legend_loc='upper left')
    return p.tolist(), r.tolist()



def plot_cm(labels, predictions, r=0.5):
    """"Plots the convusion matric from labels and predictions. 

    Parameters:
        labels (array like object): Correct labels
        predictions (array like object): Predicted values. 
        r (float): The threshold
    Returns:
        dict: Dictionary with the number of true negatives, false positives, 
        false negatives and true positives at the threhold, which is also stored.
    """
    cm = confusion_matrix(labels, predictions > r)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(r))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    out = {
        'tn': cm[0][0], 
        'fp': cm[0][1], 
        'fn': cm[1][0], 
        'tp': cm[1][1], 
        'threshold': r, 
    }
    return out


def multi_plot(xs, ys, x_label, y_label, ledgends, filename = None, baseline_val = None, random_label = 'Baseline', legend_loc = 'lower right'):
    """ Plots multiple curves
    """
    x_lim = [0, 1]
    y_lim = [0, 1]
    plt.figure()
    for i, (ledgend, x, y) in enumerate(zip(ledgends, xs, ys)):
        plt.plot(x, y, linewidth=2, label = ledgend)
        
        
    if baseline_val is not None:
        plt.plot(baseline_val[0], baseline_val[1], linewidth=2, label = random_label, color = 'red')

    label_size = 16
    plt.xlabel(x_label, size=label_size)
    plt.ylabel(y_label, size=label_size)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc=legend_loc, fontsize= 14)
    ax = plt.gca()
    if x_lim == y_lim: ax.set_aspect('equal')
    if filename is not None: plt.savefig('./Figures/'+filename+'.png', bbox_inches='tight')
    else: plt.show()

def plot_from_files(paths, ledgends):
    """ Reads from files and plots mulitple PR and ROC curves. 
    """
    datas = []
    for path in paths:
        with open(path, 'rb') as fp:
            datas.append(pickle.load(fp))
        
    fprs = []
    tprs = []
    recalls = []
    pres = []
    for data in datas:
        fprs.append(data['test_hist']['roc-curve']['fpr'])
        tprs.append(data['test_hist']['roc-curve']['tpr'])
        recalls.append(data['test_hist']['pr-curve']['recall'])
        pres.append(data['test_hist']['pr-curve']['precission'])
    multi_plot(fprs, tprs, 'False positive rate', 'True positive rate', ledgends, filename = None)
    multi_plot(recalls, pres, 'Recall', 'Precision', ledgends, legend_loc='lower left', filename= None)

# example of how to plot mulitple ROC and precision-recall curves from saved output files
#paths = ['./training_out/best_lstm_7.p', './training_out/best_lstm_10.p'] # change to correct file names
#ledgends = ['7', '10'] # changed to desired ledgends
#plot_from_files(paths, ledgends)