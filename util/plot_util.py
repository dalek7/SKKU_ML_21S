
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PlotSVM2D(X, y, model, title="SVM", xmin=-2, xmax=2, ymin=-2, ymax=2):
    import matplotlib as mpl
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                         np.arange(ymin, ymax, (ymax-ymin)/1000))
    ZZ = np.reshape(model.predict(
        np.array([XX.ravel(), YY.ravel()]).T), XX.shape)
    fig = plt.figure(figsize=(5,5))
    plt.contourf(XX, YY, ZZ, cmap=mpl.cm.Paired_r, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xlabel("x0")
    plt.ylabel("x1")


def PlotSVM3D(X, Y, model):
    import numpy as np
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    w = model.coef_
    b = model.intercept_

    x1, x2 = np.meshgrid(X1, X2)
    x3 = -(w[0][0]*x1 + w[0][1]*x2 + b) / w[0][2]

    fig = plt.figure()
    axes2 = fig.add_subplot(111, projection = '3d')
    axes2.scatter(X1, X2, X3, c = Y)
    axes1 = fig.gca(projection = '3d')
    axes1.plot_surface(x1, x2, x3, alpha = 0.01)
    #plt.show()
    
    
def plot_confusion_matrix(actual, predicted, classes, title='Confusion Matrix', normalize=False, figsize=(8, 8),
                           dpi=72, cmap=plt.cm.binary):
    import pandas as pd
    
    if not normalize:
        conf_matrix = pd.crosstab(actual, predicted)  # confusion_matrix(actual, predicted)
    else:
        conf_matrix = pd.crosstab(actual, predicted).apply(lambda r: r / r.sum(), axis=1)

    #classes = ['c0', 'c1']
    #classes = ['c{}'.format(i) for i in range(n_classes)]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)  # , cmap=plt.cm.Greens # plt.cm.viridis
    plt.title(title, size=12)
    plt.colorbar(fraction=0.041, pad=0.05)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    '''
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
        horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")
    '''
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    plt.tight_layout()
    return fig