'''

'''
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

def PICP(s: np.array, y: np.array, q:float=0.95) -> float: 
    '''
    Prediction Interval Coverage Probability (PICP)

    >>> Test_PICP()
    True

    .. math::
    Given an input $\mathbf{x}^{(i)},$ a prediction interval $\left[\hat{y}_{L}^{(i)}, \hat{y}_{U}^{(i)}\right]$ of a sample $i$ captures the future observation (target variable) $y^{(i)}$ with the probability equal or greater than $\gamma \in[0,1]$ (eq. 1). The value of $\gamma$ is commonly set to 0.95 or 0.99 . Common in the literature is an alternative notation with
    $\boldsymbol{\alpha}$
    $$
    \operatorname{Pr}\left(\hat{y}_{L}^{(i)} \leq y^{(i)} \leq \hat{y}_{U}^{(i)}\right) \geq \gamma=(1-\alpha)
    $$
    Given $n$ samples, the quality of the generated prediction intervals is assessed by measuring the prediction interval coverage probability (PICP)
    $$
    P I C P=\frac{c}{n}
    $$
    where
    $$
    c=\sum_{i=1}^{n} k_{i}
    $$
    for
    $$
    k_{i}=\left\{\begin{array}{ll}
    1 & \text { if } \hat{y}_{L}^{(i)} \leq y^{(i)} \leq \hat{y}_{U}^{(i)} \\
    0 & \text { otherwise }
    \end{array}\right.
    $$

    It is desired to achieve PICP ≥ q. 

    1. Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles. [cited 2020 Aug 16]. Available from: https://arxiv.org/abs/2007.09670


    Args:
        s (np.array): represents predicted samples, shape (n, S). S represents the number of samples predicted for each observation. 
        y (np.array): represents true predicted, shape (n,). 

    Returns:
        float: PICP metric 
    '''
    assert (q >= 0.) & (q <= 1.), 'prediction interval provided (q) is not within bounds [0,1]'

    Yl, Yu = np.quantile(s, q=[(1-q)/2, (1+q)/2], axis=1)
    k = (y > Yl) * (y < Yu)
    c = np.sum(k)
    n = y.shape[0]
    PICP = c/n
    return PICP

def Test_PICP(): 
    ''' 
    '''
    np.random.seed(0)
    q = [0.5, 0.75, 0.9, 0.99]
    o = [] 
    for _q in q: 
        np.random.seed(0)
        x = np.linspace(0,2*3.14, 5000)
        y = np.sin(x) + np.random.normal(0,1,size=x.shape[0])
        s = np.array([np.sin(x) + np.random.normal(0,1,size=x.shape[0]) for _ in range(1000)]).T
        val = PICP(s, y, q=_q)
        o.append(val)
    return np.allclose(q,o, atol=0.01, rtol=0)


def MPIW(s: np.array, q:float=0.95) -> float: 
    '''
    Mean Prediction Interval Width 

    >>> Test_MPIW()
    True

    ..math::
    $$ M P I W=\frac{1}{n} \sum_{i=1}^{n} \hat{y}_{U}^{(i)}-\hat{y}_{L}^{(i)}$$

    It is desired to have MPIW as small as possible. 

    1. Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles. [cited 2020 Aug 16]. Available from: https://arxiv.org/abs/2007.09670

    Args: 
        s (np.array): predicted samples, shape (n, S), where S is number of samples drawn. 
        q (float)   : prediction interval (default=0.95)
    Returns: 
        float: MPIW
    '''
    assert (q >= 0.) & (q <= 1.), 'prediction interval provided (q) is not within bounds [0,1]'

    Yl, Yu = np.quantile(s, q=[(1-q)/2, (1+q)/2], axis=1)
    n = s.shape[0]
    MPIW = np.sum(Yu-Yl)/n
    return MPIW

def Test_MPIW(): 
    ''' 
    '''
    np.random.seed(0)
    q = [0.5, 0.75, 0.9, 0.99]
    o = [] 
    true_mpiw = []
    for _q in q: 
        np.random.seed(0)
        _il, _iu = norm(0,1).interval(_q)
        _mpiw = _iu-_il
        true_mpiw.append(_mpiw)
        x = np.linspace(0,2*3.14, 5000)
        y = np.sin(x) + np.random.normal(0,1,size=x.shape[0])
        s = np.array([np.sin(x) + np.random.normal(0,1,size=x.shape[0]) for _ in range(1000)]).T
        val = MPIW(s, q=_q)
        o.append(val)

    return np.allclose(true_mpiw, o, atol=0.1, rtol=0)

def NMPIW(s: np.array, y:np.array, q:float=0.95) -> float: 
    '''
    Normalized Mean Prediction Interval Width 

    ..math::
    $$
    N M P I W=\frac{M P I W}{r}
    $$
    where $r=\max (y)-\min (y) .$ 

    It is desired to have MPIW as small as possible. 

    1. Prediction Intervals: Split Normal Mixture from Quality-Driven Deep Ensembles. [cited 2020 Aug 16]. Available from: https://arxiv.org/abs/2007.09670

    Args: 
        s (np.array): predicted samples, shape (n, S), where S is number of samples drawn. 
        y (np.array): true predictons, shape (n,)
        q (float)   : prediction interval (default=0.95)
    Returns: 
        float: MPIW
    '''
    mpiw = MPIW(s, q)
    r = np.max(y) - np.min(y)
    return mpiw/r

def absolute_calibration_curve_auc(s: np.array, y:np.array, nq=25, plot=False) -> float:
    '''
    Scalar metric representing calibration curve performance: 
    1. calculate calibration quantile curve 
    2. take the difference with the off diagonal (q=qhat)
    3. take the absolute value 
    4. calculate the auc 

    [0,1] small value represents perfectly calibrated model. 

    Args: 
        s (np.array): predicted samples, shape (n, S), where S is number of samples drawn. 
        y (np.array): true predictons, shape (n,)
        nq (int)    : number of points to evaluate the calibation curve at. larger values will give better integration estimates. 
    Returns: 
        float: area between the quantile curve and the diagonal

    TODO: doctesting 
    '''
    q = np.linspace(0,1, nq)
    picp = np.array([PICP(s,y,q=_q) for _q in q])
    diff = np.abs(picp-q)
    delta = np.diff(q)
    auc = np.sum( delta*diff[0:-1] )

    if plot: 
        plt.figure()
        plt.plot(q,picp, 'r--', label='quanile calib.')
        plt.plot(q,q, 'g-', alpha=0.2, label='true')
        plt.xlabel('true quantile')
        plt.ylabel('predicted quantile')
        plt.fill_between(q,picp, q, alpha=0.1, label='auc')
        plt.plot(q,diff, 'b--', label='abs. diff.')
        plt.bar(x=q[:-1], height=diff[:-1], width=delta, label='integration', color='b', alpha=0.2)
        plt.legend()
        plt.show()

    return auc



if __name__ == '__main__': 

    #print('passed PICP test:', Test_PICP())
    #print('passed MPIW test:', Test_MPIW())

    np.random.seed(0)
    x = np.linspace(0,2*3.14, 5000)
    y = np.sin(x) + np.random.normal(0,1,size=x.shape[0])
    s = np.array([np.sin(x) + np.random.normal(-0.2,5,size=x.shape[0]) for _ in range(10)]).T
    auc = absolute_calibration_curve_auc(s,y, plot=True)
    print('auc:', auc)