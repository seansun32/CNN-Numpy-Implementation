import numpy as np



def sgd(w,dw,optim_config):
    learning_rate=optim_config.get('learning_rate',1e-3)
    w -= learning_rate*dw

    return w,optim_config



def adam(w,dw,config):
    pass



def adagram(w,dw,config):
    pass
