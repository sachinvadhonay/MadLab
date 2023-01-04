import numpy as np

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100

class NeuralNetwork(object):
    def __init__(self):
        self.inputsize=2
        self.outputsize=1
        self.hiddensize=3
        self.w1=np.random.rand(self.inputsize,self.hiddensize)
        self.w2=np.random.rand(self.hiddensize,self.outputsize)

    def feedfarward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        output=self.sigmoid(self.z3)
        return output
    
    def sigmoid(self,s,derive=False):
        if derive==True:
            return s*(1-s)
        return 1/(1+np.exp(-s))


    def bacward(self,x,y,output):
        self.output_error=y-output
        self.output_data=self.output_error*self.sigmoid(output,derive=True)
        self.z2_error=self.output_data.dot(self.w2.T)
        self.z2_data=self.z2_error*self.sigmoid(self.z2,derive=True)
        self.w1+=x.T.dot(self.z2_data)
        self.w2+=self.z2.T.dot(self.output_data)

    def train(self,x,y):
        output=self.feedfarward(x)
        self.bacward(x,y,output)

nn=NeuralNetwork()
for i in range(500000):
    if i%100==0:
        print("loss:",np.mean(np.square(nn.feedfarward(x))))
    nn.train(x,y)

print("input:",x)
print("output:",y)
print("pridicted output:",nn.feedfarward(x))
print("loss:",np.mean(np.square(nn.feedfarward(x))))
