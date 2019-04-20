# DasNet
DasNet is a new multi-task learning framework to accelerate inference speed.
In this project we provide DasNet framework and apply this method on ImageNet and Birds200. Dataset also includes MIT67 and CatVsDog. 
To demonstrate the accuracy of our approach, we compared the recent popular pruning methods to a baseline. Specifically, the comparison 
algorithm has Taylor prune(Pruning convolutionalneuralnetworksforresourceefﬁcienttransferlearning), ThiNet( Thinet: A ﬁlter level pruning method for deep neural network compres
sion) and AutoPruner(Autopruner: An end-to-end trainable ﬁlter pruning method for efﬁcient deep model inference). Figure 1 and 2 shows our results.
![figure 1](https://github.com/pangxiao201314/DasNet/blob/master/table1.png)
![figure 2](https://github.com/pangxiao201314/DasNet/blob/master/figure2.png)
From the Figure 1 we can see that the experimental results of the all methods are similar,but the DasNet is slightly better. In addition to inference parameter efﬁciency, another direct evaluation criterion is the amount of ﬂoating-point operations, which is a popular metric to evaluate the complexity of CNN models. Figure 2 shows the ﬂoating-point calculations of DasNet and other 3 pruning methods while using different sized network. Dataset used is CatVsDog. Our method improves floating-point operations better.
![figure 3](https://github.com/pangxiao201314/DasNet/blob/master/figure3.png)
