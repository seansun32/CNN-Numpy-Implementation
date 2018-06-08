import numpy as np
from layers import *
import optimizer



class CNNModel(object):
    def __init__(self,weight_scale=1e-2,reg=0.0,learning_rate=1e-2,rate_decay=0.95):
        self.weight_scale=weight_scale
        self.learning_rate=learning_rate
        self.reg=reg
        self.rate_decay=rate_decay
        self.conv_num=0
        self.fc_num=0
        self.params={}
        self.configs={}
        #cache_history=[]

    def conv_layer(self,x,filter_param,pool_param):
        '''
        x:(N,C,H,W)
        w:(F,C,HH,WW)
        b:(F,)
        '''

        filter_size,filter_num,stride,pad=filter_param
        pool_size,pool_stride=pool_param

        self.conv_num += 1
        channel_num=x.shape[1] #get channel number
        conv_config={'stride':stride,'pad':pad}
        
        self.params['conv_W%d'%(self.conv_num)]=self.weight_scale*np.random.randn(filter_num,channel_num,filter_size,filter_size)
        self.params['conv_b%d'%(self.conv_num)]=np.zeros(filter_num)
        self.configs['conv_config%d'%(self.conv_num)]=conv_config

        pool_config={'pool_weight':pool_size,'pool_height':pool_size,'stride':pool_stride}
        self.configs['pool_config%d'%(self.conv_num)]=pool_config


    
        
        w=self.params['conv_W%d'%(self.conv_num)]   
        b=self.params['conv_b%d'%(self.conv_num)]

        out,cache=conv_relu_pool_forward(x,w,b,conv_config,pool_config)
        #self.cache_history.append(cache) 

        return out
        

    '''
    def maxpool_layer(self,x,pool_size,stride):
        pool_config={'pool_width':pool_size,'pool_height':pool_size,'stride':stride}
        self.params['pool_config%d'%(self.conv_num)]=pool_config

        out,cache=maxpool_forward(x,pool_config)
        self.cache_history.append(cache) 

        return out
    '''

    def fc_layer(self,x,out_dim):
        self.fc_num += 1
        N=x.shape[0]
        in_dim=np.reshape(x,(N,-1)).shape[1]
        
        self.params['fc_W%d'%(self.fc_num)]=self.weight_scale*np.random.randn(in_dim,out_dim)
        self.params['fc_b%d'%(self.fc_num)]=np.zeros(out_dim)

        
        w=self.params['fc_W%d'%(self.fc_num)]
        b=self.params['fc_b%d'%(self.fc_num)]

        out,cache=fc_relu_forward(x,w,b)
        #self.cache_history.append(cache)

        return out
        

    def loss(self,x,y=None):
        #pass
        out=x
        L2reg=0
        grads={}
        cache_history=[]


        #forward
        for i in range(self.conv_num):
            w=self.params['conv_W%d'%(i+1)]   
            b=self.params['conv_b%d'%(i+1)]
            conv_config=self.configs['conv_config%d'%(i+1)]
            pool_config=self.configs['pool_config%d'%(i+1)]

            out,cache=conv_relu_pool_forward(out,w,b,conv_config,pool_config)
            cache_history.append(cache)
            L2reg += np.sum(self.params['conv_W%d'%(i+1)]**2)
        
        cross_shape=out.shape

        for i in range(self.fc_num-1):
            w=self.params['fc_W%d'%(i+1)]
            b=self.params['fc_b%d'%(i+1)]
            out,cache=fc_relu_forward(out,w,b) 
            cache_history.append(cache)
            L2reg += np.sum(self.params['fc_W%d'%(i+1)]**2)

        if 'dropout' in self.params:
            pass

        i += 1
        scores,cache=fc_forward(out,self.params['fc_W%d'%(i+1)],self.params['fc_b%d'%(i+1)])
        cache_history.append(cache)

        if y is None:
            return scores 

        loss,dout=softmax_loss(scores,y)
        loss += 0.5*self.reg*L2reg


        #backward
        dout,grads['fc_W%d'%(i+1)],grads['fc_b%d'%(i+1)]=fc_backward(dout,cache_history.pop())
        grads['fc_W%d'%(i+1)] += self.reg*self.params['fc_W%d'%(i+1)]
        

        for i in reversed(range(self.fc_num-1)):
            if 'dropout' in self.params:
                pass

            dout,grads['fc_W%d'%(i+1)],grads['fc_b%d'%(i+1)]=fc_relu_backward(dout,cache_history.pop())
            grads['fc_W%d'%(i+1)] += self.reg*self.params['fc_W%d'%(i+1)]
        
        #--------caution here-----------
        dout=np.reshape(dout,cross_shape)
        #-------------------------------

        for i in reversed(range(self.conv_num)):
            dout,grads['conv_W%d'%(i+1)],grads['conv_b%d'%(i+1)]=conv_relu_pool_backward(dout,cache_history.pop())
            grads['conv_W%d'%(i+1)] += self.reg*self.params['conv_W%d'%(i+1)]
        
        return loss,grads
    



    def evaluate(self,x,y,num_samples=None):
        num_valid=x.shape[0]
        if num_samples is not None:
            mask=np.random.choice(num_valid,num_samples)
            x=x[mask]
            y=y[mask]

        scores=self.loss(x)
        y_pred=np.argmax(scores,axis=1)
        #print('y_pred shape: ',y_pred.shape)
        #print('y shape: ',y.shape)
        #print(y_pred[:5])
        #print('---------')
        #print(y[:5])
        acc=np.mean(y_pred==y)

        return acc

    def train(self,x_train,y_train,x_val,y_val,batch_size,epoch,update_rule):
        #pass
        num_train=x_train.shape[0]
        iterations_per_epoch=np.maximum(1,num_train//batch_size)

        train_acc_history=[]
        val_acc_history=[]
        loss_history=[]
        num_iterations=iterations_per_epoch*epoch

        for i in range(num_iterations):
            mini_batch_idx=np.random.choice(num_train,batch_size)
            x_batch=x_train[mini_batch_idx]
            y_batch=y_train[mini_batch_idx]
            loss,grads=self.loss(x_batch,y_batch)
            loss_history.append(loss)
            
            if update_rule=='sgd':
                config={'learning_rate':self.learning_rate}
                for k,w in self.params.items():
                    dw=grads[k]
                    #print(dw)
                    #print('--------')
                    self.params[k] -= self.learning_rate*dw
                    #next_w,next_config=optimizer.sgd(w,dw,config)
                    #self.params[k]=next_w
            
            if i%100==0:
                print('(Iterations:%d/%d) loss:%f'%(i,num_iterations,loss_history[-1]))
            
            if i%iterations_per_epoch==0:
                self.learning_rate *= self.rate_decay
                epoch_idx=i/iterations_per_epoch

                train_acc=self.evaluate(x_train,y_train,num_samples=100)
                val_acc=self.evaluate(x_val,y_val,num_samples=100)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                print('(Epoch %d/%d), train_acc:%f   val_acc:%f'%(epoch_idx,epoch,train_acc,val_acc))

    def predict(self,x):
        #pass
        scores=self.loss(x)
        y_pred=np.argmax(x)

        return y_pred


