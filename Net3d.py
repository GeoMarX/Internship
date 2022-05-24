
import jax
import jax.numpy as jnp
import haiku as hk


class SpectralConv3(hk.Module):
  """SpectralConv3d in Jax"""

  def __init__(self, in_channels=1, out_channels=1, modes1=32, modes2=32, modes3=32, is_training=True, name='Layer'):

    super().__init__(name=name)
    self.name= name
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
    self.modes2 = modes2
    self.modes3 = modes3
    
    self.scale = (1 / (self.in_channels * self.out_channels))
    self.weights1= hk.get_parameter(str(self.name)+ "w1", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights2= hk.get_parameter(str(self.name)+ "w2", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights3= hk.get_parameter(str(self.name)+ "w3", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    self.weights4= hk.get_parameter(str(self.name)+ "w4", shape=[in_channels, out_channels, self.modes1, self.modes2, self.modes3], dtype=float, init=hk.initializers.VarianceScaling())
    print("Spectral Initialized")


  def __call__(self, pot_k):

    def compl_mul3d(input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return jnp.einsum("bixyz,ioxyz->boxyz", input, weights)
    
  
        
    self.weights1=self.weights1*self.scale
    self.weights2=self.weights2*self.scale
    self.weights3=self.weights3*self.scale
    self.weights4=self.weights4*self.scale

    batchsize=1
    
    
    #x_ft=x_ft.reshape(1,1,64,64,33)
    x_ft=pot_k
    
    _,_,dim1,dim2,dim3=x_ft.shape
    
    out_ft=jnp.zeros([batchsize, self.out_channels, dim1, dim2, dim3], dtype=float)
    out_ft=out_ft.at[:, :, :self.modes1, :self.modes2, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1))

    out_ft=out_ft.at[:, :, -self.modes1:, :self.modes2, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2))

    out_ft=out_ft.at[:, :, :self.modes1, -self.modes2:, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3))

    out_ft=out_ft.at[:, :, -self.modes1:, -self.modes2:, :self.modes3]. \
    set(compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4))

    

    return out_ft




class FNO3d(hk.Module):
  '''
  NN with Spectral Conv3D layers
  Takes as input a 3D array
  
  '''
  
    def __init__(self,width,modes1=8,modes2=8,modes3=8, padding=6,name="PaperNetwork"):
        
        super().__init__(name=name)
        self.modes1= modes1
        self.modes2= modes2
        self.modes3= modes3
        self.width = width
        self.padding=6
        self.conv0 = SpectralConv3(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l0')
        self.conv1 = SpectralConv3(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l1')
        self.conv2 = SpectralConv3(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l2')
        self.conv3 = SpectralConv3(self.width, self.width, self.modes1, self.modes2, self.modes3, name='l3')
        self.w0 = hk.Conv3D(self.width, 1)
        self.w1 = hk.Conv3D(self.width,  1)
        self.w2 = hk.Conv3D(self.width,  1)
        self.w3 = hk.Conv3D(self.width,  1)
        self.w4 = hk.Conv3D(3,  1)
        
    def __call__(self,pot_k):
      
        
        x=pot_k
        
        x = jax.numpy.pad(x, [0,self.padding],mode='wrap')
        dim1,dim2,dim3=x.shape
        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1=x1.reshape(1,1,dim1,dim2,int(dim3/2+1))
        x1 = self.conv0(x1)
        x1=x1.reshape(self.width,dim1,dim2,int(dim3/2+1))
        x1=jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        
        #Real Space
        x=x.reshape(dim1,dim2,dim3,1)
        x2 = self.w0(x)
        x2=x2.reshape(1,self.width,dim1,dim2,dim3)
        # print("Fourier shape: ", x1.shape ,"Conv3d shape: ", x2.shape)
        x = x1 + x2
        x = jax.nn.gelu(x)
        
        
        #Fourier Space
        x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        x1 = x1.reshape(1,self.width,dim1,dim2,int(dim3/2+1))
        x1 = self.conv1(x1)
        x1 = x1.reshape(self.width,dim1,dim2,int(dim3/2+1))
        x1 = jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        
        #Real Space
        x = x.reshape(dim1,dim2,dim3,self.width)
        x2= self.w1(x)
        x2= x2.reshape(1,self.width,dim1,dim2,dim3)
        # print("Fourier shape: ", x1.shape ,"Conv3d shape: ", x2.shape)
        x = x1 + x2
        x = jax.nn.gelu(x)
        
        
        # # #Fourier Space
        # x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        # x1 = x1.reshape(1,self.width,dim1,dim2,int(dim3/2+1))
        # x1 = self.conv2(x1)
        # x1 = x1.reshape(self.width,dim1,dim2,int(dim3/2+1))
        # x1 = jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        
        # #Real Space
        # x = x.reshape(dim1,dim2,dim3,self.width)
        # x2= self.w2(x)
        # x2= x2.reshape(1,self.width,dim1,dim2,dim3)
        # # print("Fourier shape: ", x1.shape ,"Conv3d shape: ", x2.shape)
        # x = x1 + x2
        # x = jax.nn.gelu(x)
        
        
        # #Fourier Space
        # x1 = jnp.fft.rfftn(x,s=(dim1,dim2,dim3))
        # x1 = x1.reshape(1,self.width,dim1,dim2,int(dim3/2+1))
        # x1 = self.conv3(x1)
        # x1 = x1.reshape(self.width,dim1,dim2,int(dim3/2+1))
        # x1 = jnp.fft.irfftn(x1,s=(dim1,dim2,dim3))
        
        # #Real Space
        # x = x.reshape(dim1,dim2,dim3,self.width)
        # x2= self.w3(x)
        # x2= x2.reshape(1,self.width,dim1,dim2,dim3)
        # # print("Fourier shape: ", x1.shape ,"Conv3d shape: ", x2.shape)
        # x = x1 + x2
        # x = jax.nn.gelu(x)

        x=x.reshape(self.width,dim1,dim2,dim3)
        x = x[ :-self.padding, :-self.padding, :-self.padding, :-self.padding]

    
        dim1=dim1-self.padding
        dim2=dim2-self.padding
        dim3=dim3-self.padding

        #Final Convolution
        x=x.reshape(dim1,dim2,dim3,self.width-self.padding)
        x = self.w4(x)
        #x=x.reshape(1,1,64,64,64)
        # print("Before: ", x.shape)
        

        

        final=1
        x=x.reshape(3,dim1,dim2,dim3,final) 

        
        mod = hk.Reshape(output_shape=(-1, final))
        x=mod(x)
        #print(mod(x).shape)
        return x.reshape(dim1*dim2*dim3,3)
