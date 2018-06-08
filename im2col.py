import numpy as np

def im2col_index(x_shape,HH,WW,pad,stride):
    N,C,H,W=x_shape

    H_out=1+(H-HH+2*pad)//stride
    W_out=1+(W-WW+2*pad)//stride
    
    #row index
    r0=np.repeat(np.arange(HH),WW)
    r0=np.tile(r0,C)
    r_bias=stride*np.repeat(np.arange(H_out),W_out)
    r=r0.reshape(-1,1)+r_bias.reshape(1,-1)
    
    #col index
    c0=np.tile(np.arange(WW),HH*C)
    c_bias=stride*np.tile(np.arange(W_out),H_out)
    c=c0.reshape(-1,1)+c_bias.reshape(1,-1)
    
    #channel index
    d=np.repeat(np.arange(C),HH*WW).reshape(-1,1)

    return (r,c,d)


def im2col(x,HH,WW,pad,stride):
    #x:N,C,H,W    not H,W,D,N
    x_padded=None

    x_padded=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    r,c,d=im2col_index(x.shape,HH,WW,pad,stride)
    cols=x_padded[:,d,r,c]
    #print(cols.shape)
    cols=np.concatenate(cols,axis=1)

    return cols

def im2col1(x, field_height, field_width, pad=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    i, j, k = im2col_index(x.shape, field_height, field_width, pad,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols



def col2im(cols,x_shape,HH,WW,pad,stride):
    N,C,H,W=x_shape
    H_padded=0
    W_padded=0

    H_padded,W_padded=H+2*pad,W+2*pad
    x_padded=np.zeros((N,C,H_padded,W_padded))

    r,c,d=im2col_index(x_shape,HH,WW,pad,stride)
    #print('col1:',cols.shape) 
    cols_reshaped=cols.reshape((HH*WW*C),-1,N)
    #col_reshaped=np.split(cols,axis=1)
    #print('col_shape:',cols_reshaped.shape) 
    cols_reshaped=cols_reshaped.transpose(2,0,1)
    #x_padded=x_padded.transpose(2,3,1,0)
    np.add.at(x_padded, (slice(None), d, r, c), cols_reshaped)
        
    #print('x_padded:',x_padded)
    #np.add.at(x_padded,(r,c,d,slice(None)),cols_reshaped)
    #print('x_pad1:',x_padded.shape) 
    #x_padded=x_padded.transpose(3,2,0,1)
    #print('xpad:',x_padded.shape)
    if pad==0:
        return x_padded
    return x_padded[:,:,pad:-pad,pad:-pad]

'''
def col2im(cols, x_shape, field_height, field_width, padding,stride):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    #k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
    #                             stride)
    i,j,k=im2col_index(x_shape,field_height,field_width,padding,stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    print(x_padded)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
'''


def conv_forward_fast(x,w,b,conv_param):
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape

    stride=conv_param.get('stride')
    pad=conv_param.get('pad')
    
    H_out=1+(H-HH+2*pad)//stride
    W_out=1+(W-WW+2*pad)//stride

    #x_pad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
    out=np.zeros((N,F,H_out,W_out))

    x_col=im2col(x,HH,WW,pad,stride)
    w_col=np.reshape(w,(F,-1))
    
    output_col=w_col.dot(x_col)+b.reshape(-1,1)
    #print('ouputshape:',output_col.shape)
    output_col=output_col.reshape((F,N,H_out,W_out))
    output_col=output_col.transpose(1,0,2,3)
    cache=(x,w,b,x_col,w_col,conv_param)

    return output_col,cache


def conv_backward_fast(dout,cache):
    
    x,w,b,x_col,w_col,conv_param=cache
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    stride=conv_param.get('stride',1)
    pad=conv_param.get('pad',0)

    #H_out=1+(H-HH+2*pad)//stride
    #W_out=1+(W-WW+2*pad)//stride
    
    #print(dout.shape)
    #print('dout:',dout)

    #dout=dout.transpose(1,0,2,3)
    #print('dout:=====',dout.shape)
    dout_x=dout.transpose(1,2,3,0)
    dout_x=np.reshape(dout_x,(F,-1))
    dx_col=w_col.T.dot(dout_x)

    dout_w=dout.transpose(1,0,2,3)
    dout_w=np.reshape(dout_w,(F,-1))
    dw_col=dout_w.dot(x_col.T)
    db=np.sum(dout_w,axis=1)

    #print(dx_col)
    #print(dx_col.shape)
    #print('---------')
    
    dx=col2im(dx_col,x.shape,HH,WW,pad,stride)
    dw=np.reshape(dw_col,(F,C,HH,WW))
    #print(dx) 

    return dx,dw,db


def maxpool_forward_fast(x,pool_param):
    N,C,H,W=x.shape
    HH,WW=pool_param['pool_height'],pool_param['pool_weight']
    stride=pool_param['stride']

    H_out=1+(H-HH)//stride
    W_out=1+(W-WW)//stride
    
    #x=x.transpose()
    '''
    '''
    #x_split=x.reshape(N*C,H,W,1)
    #x_split=x.transpose(0,3,1,2)
    ''' 
    '''
    x_split=x.reshape(N*C,1,H,W)
    x_cols=im2col1(x_split,HH,WW,pad=0,stride=stride)
    #print('1:',x_cols.shape)
    x_cols_argmax=np.argmax(x_cols,axis=0)
    #print(x_cols_argmax)
    x_cols_max=x_cols[x_cols_argmax,np.arange(x_cols.shape[1])]
    #print('col_maxshape:',x_cols_max.shape) 

    out=x_cols_max.reshape(H_out,W_out,N,C).transpose(2,3,0,1)
    cache=(x,x_cols,x_cols_argmax,pool_param)
    #print('out:',out)
    #print(out.shape)

    return out,cache


def maxpool_backward_fast(dout,cache):
    x,x_cols,x_cols_argmax,pool_param=cache
    N,C,H,W=x.shape
    HH,WW=pool_param['pool_height'],pool_param['pool_weight']
    stride=pool_param['stride']

    dout_reshaped=dout.transpose(2,3,0,1).flatten()
    dx_cols=np.zeros_like(x_cols)
    dx_cols[x_cols_argmax,np.arange(dx_cols.shape[1])]=dout_reshaped

    dx=col2im(dx_cols,(N*C,1,H,W),HH,WW,pad=0,stride=stride)
    dx=dx.reshape(x.shape)

    return dx


