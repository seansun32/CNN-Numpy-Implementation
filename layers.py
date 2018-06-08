import numpy as numpy
from im2col import *

#--------------basic layer-------------------
def fc_forward(x,w,b):
    '''
    x:(N,D1,D2,..Dk)
    w:(H,C)
    b:(C,)
    out:(N,C)
    '''
    #number of input 
    N=x.shape[0]
    x_flat=np.reshape(x,(N,-1))
    assert(x_flat.shape[1]==w.shape[0])

    out=x_flat.dot(w)+b
    cache=(x,w,b)

    return out,cache 



def fc_backward(dout,cache):
    '''
    dout:(N,C)
    dw:(H,C)
    db:(C)
    dx_flat:(N,H)
    '''

    x,w,b=cache
    N=x.shape[0]
    x_flat=np.reshape(x,(N,-1))

    dw=x_flat.T.dot(dout)
    dx_flat=dout.dot(w.T)
    dx=np.reshape(dx_flat,x.shape)
    db=np.sum(dout,axis=0)

    return dx,dw,db


def relu_forward(x):
    mask=x>0
    out=x*mask

    cache=(x,mask)

    return out,cache


def relu_backward(dout,cache):
    x,mask=cache
    dx=dout*mask

    return dx
   

def conv_forward(x,w,b,conv_param): 
    '''
    x:(N,C,H,W)
    w:(F,C,HH,WW)
    b:(F)
    '''
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape

    stride=conv_param.get('stride',1)
    pad=conv_param.get('pad',0)

    H_out=1+(H-HH+2*pad)//stride
    W_out=1+(W-WW+2*pad)//stride

    x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
    out=np.zeros(N,F,H_out,W_out)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n,f,i,j]=np.sum(x_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW]*w[f,:,:,:])+b[f]
    
    cache=(x,w,b,conv_param)
    return out,cache

def conv_backward(dout,cache):
    '''
    dout:(N,F,H_out,W_out)
    '''
    x,w,b,conv_param=cache

    N,C,H,W=x.shape
    F,_,HH,WW=w.shape

    stride=conv_param.get('stride',1)
    pad=conv_param.get('pad',0)

    H_out=1+(H-HH+2*pad)//stride
    W_out=1+(W-WW+2*pad)//stride
    
    x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
    dx_pad=np.zeros(x_pad.shape)
    dw=np.zeros(w.shape)
    db=np.zeros(b.shape)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += dout[n,f,i,j]*w[n,:,:,:]
                    dw[f,:,:,:] += dout[n,f,i,j]*x_pad[n,:,i*stride+HH,j*stride+WW]
                    db[f] += dout[n,f,i,j]
    
    dx=dx_pad[:,:,pad:pad+H,pad:pad+W]

    return dx,dw,db


def maxpool_forward(x,pool_param):
    N,C,H,W=x.shape

    stride=pool_param.get('stride',2)
    HH=pool_param.get('pool_height',2)
    WW=pool_param.get('pool_width',2)

    H_out=1+(H-HH)//stride
    W_out=1+(W-WW)//stride

    out=np.zeros((N,C,H_out,W_out))

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[n,c,i,j]=np.max(x[n,c,i*stride:i*stride+HH,j*stride:j*stride+WW])

    cache=(x,pool_param)
    return out,cache


def maxpool_backward(dout,cache):
    x,pool_param=cache
    N,C,H,W=x.shape
    
    stride=pool_param.get('stride',2)
    HH=pool_param.get('pool_height',2)
    WW=pool_param.get('pool_width',2)

    H_out=1+(H-HH)//stride
    W_out=1+(W-WW)//stride

    for n in range(N):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    ind_H,ind_W=np.unravel_index(np.argmax(x[n,c,i*stride:i*stride+HH,j*stride:j*stride+WW]),(HH,WW))
                    dx[n,c,i*stride+ind_H,j*stride+inde_W]=dout[n,c,i,j]

    return dx



def softmax(x,y):
    pass


def softmax_loss(x,y):
    '''
    x:(N,C)
    y:(N)

    return loss,dout
    '''
    
    N,C=x.shape
    x -= np.max(x,axis=1,keepdims=True)
    correct_x=x[np.arange(N),y]
    
    prob=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    correct_prob=np.exp(correct_x)/np.sum(np.exp(x),axis=1)
    loss=np.sum(-1*np.log(correct_prob),axis=0)
    loss /= N

    dout=prob;
    dout[np.arange(N),y] -= 1
    dout /= N

    return loss,dout



#-----------------sandwitch layer--------------------------------------------
'''
here we use fast convolution & fast maxpooling which implemented in im2col.py
because implementation above uses too many loops and thus time&resource comsuming
'''

#fc_relu
def fc_relu_forward(x,w,b):
    out,cache_fc=fc_forward(x,w,b)
    out,cache_relu=relu_forward(out)
    
    cache=(cache_fc,cache_relu)
    return out,cache
    
def fc_relu_backward(dout,cache):
    cache_fc,cache_relu=cache
    dout=relu_backward(dout,cache_relu)
    dout,dw,db=fc_backward(dout,cache_fc)

    return dout,dw,db



#conv+relu
def conv_relu_forward(x,w,b,conv_param):
    out,cache_conv=conv_forward_fast(x,w,b,conv_param)
    out,cache_relu=relu_forward(out)

    cache=(cache_conv,cache_relu)
    return out,cache

def conv_relu_backward(dout,cache):
    cache_conv,cache_relu=cache

    dout=relu_backward(dout,cache_relu)
    dout,dw,db=conv_backward_fast(dout,cache_conv)

    return dout,dw,db


#conv+relu+maxpool
def conv_relu_pool_forward(x,w,b,conv_param,pool_param):
    out,cache_conv=conv_forward_fast(x,w,b,conv_param)
    out,cache_relu=relu_forward(out)
    out,cache_pool=maxpool_forward_fast(out,pool_param)

    cache=(cache_conv,cache_relu,cache_pool)
    return out,cache

def conv_relu_pool_backward(dout,cache):
    cache_conv,cache_relu,cache_pool=cache

    dout=maxpool_backward_fast(dout,cache_pool)
    dout=relu_backward(dout,cache_relu)
    dout,dw,db=conv_backward_fast(dout,cache_conv)
    
    return dout,dw,db

