from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import csv
def square_error(y_org,y_line):   #finding the square_error of the models
    return sum((y_line-y_org)**2);
def coeff_of_det(y_org,y_line):     #finding the co-efficient of determination that is accuracy of the model
    y_mean=[mean(y_org) for y in y_org];
    square_err_reg=square_error(y_org,y_line);
    square_err_mean=square_error(y_org,y_mean);
    return 1-(square_err_reg/square_err_mean);
def Linear_reg(m,b,x1):                                 #finding the y-intercept of the linear predictive model
    y=[(m*x+b) for x in x1];
    return np.array(y,dtype=np.float64);
def Linear_Comb1(x1,y1):             #model using covariance and variance
    co_var=0.0
    for i in range(len(x1)):
        co_var+=(x1[i]-mean(x1))*(y1[i]-mean(y1))
    var=sum(((x-mean(x1))**2) for x in x1);
    m=co_var/var;
    b=mean(y1)-m*mean(x1);
    return m,b;
def Linear_Comb2(x1,y1):         # model using normal statistical analysis
    m=(mean(x1)*mean(y1)-mean(x1*y1))/(((mean(x1))**2)-mean(x1*x1));
    b=mean(y1)-m*mean(x1);
    return m,b;
def main():
    filename='Salary_Data.csv'
    with open(filename,'r') as csvfile:
        data=csv.reader(csvfile)
        dataset=list(data)
    x1=[]
    y1=[]
    for i in range(1,len(dataset)):
        x1.append(dataset[i][0]);
    for i in range(1,len(dataset)):
        y1.append(dataset[i][1]);
    x1=np.array(x1,dtype=np.float64)
    y1=np.array(y1,dtype=np.float64)
    m=0
    b=0
    m1,b1=Linear_Comb1(x1,y1);
    m2,b2=Linear_Comb2(x1,y1);
    y_comb1=Linear_reg(m1,b1,x1);
    y_comb2=Linear_reg(m2,b2,x1);
    N=len(y1);                                                          
    for j in range(4000):                                               #model:gradient descent
        m_grad=0;
        b_grad=0;
        for i in range(N):
            m_grad+=(2/N)*((m*x1[i]+b-y1[i])*x1[i]);
            b_grad+=(2/N)*(m*x1[i]+b-y1[i]);
        m=m-(0.003*m_grad);
        b=b-(0.003*b_grad);
    y_gra=Linear_reg(m,b,x1);
    gra=coeff_of_det(y1,y_gra);
    comb1=coeff_of_det(y1,y_comb1);
    comb2=coeff_of_det(y1,y_comb2);
    print("the accuracy of first model:",comb1);                #accuracy of the covariance-variance model
    print("the accuracy of second model:",comb2);            #accuracy of the normal statistical analysis
    print("the accuracy of gradient descent:",gra);              #accuracy of the gradient descent model
    plt.scatter(x1,y1);
    plt.plot(x1,y_comb1,color='red');                           
    plt.plot(x1,y_comb2,color='orange');
    plt.plot(x1,y_comb1,color='green');
    plt.show();
main();
