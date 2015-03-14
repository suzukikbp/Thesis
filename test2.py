import numpy as np
import pylab
from sklearn.datasets import load_digits
import optunity
import optunity.metrics
from sklearn import svm, datasets


#  sample:1797
#  pixel:8x8
digits = load_digits()


xdat = digits.data
xdat /= xdat.max()
ydat = digits.target


# draw the first 10 samples
# digits.images[i] : image
# digits.target[i] : the lael of image
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:10]):
    pylab.subplot(2, 5, index + 1)
    pylab.axis('off')
    pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
    pylab.title('%i' % label)
pylab.show()



# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=xdat, y=ydat, num_folds=10, num_iter=2)
def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
    model = svm.SVC(C=C, gamma=gamma).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return optunity.metrics.accuracy(y_test, y_pred)

# perform tuning
optimal_pars, _, _ = optunity.maximize(svm_acc, num_evals=200, C=[0, 10], gamma=[0, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = svm.SVC(**optimal_pars).fit(xdat, ydat)
print('optimal parameters: ' + str(optimal_pars))





