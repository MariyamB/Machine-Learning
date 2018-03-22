import scipy.io
import numpy as np
import pylab as pl

from sklearn.linear_model import lasso_path
from sklearn.linear_model import LassoCV
from sklearn.datasets import load_boston

boston = load_boston()
model = LassoCV(cv=10,max_iter=1000).fit(boston.data, boston.target)
print('alpha', model.alpha_)
print ('coef', model.coef_)
scipy.io.savemat('boston.mat',mdict={'boston':[boston.data, boston.target]})
eps = 1e-2 # the smaller it is the longer is the path
models = lasso_path(boston.data, boston.target, eps=eps)
alphas_lasso = np.array([model.alpha for model in models])
coefs_lasso = np.array([model.coef_ for model in models])

pl.figure(1)
ax = pl.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = pl.semilogx(alphas_lasso,coefs_lasso)
pl.gca().invert_xaxis()
pl.xlabel('alpha')
pl.show()