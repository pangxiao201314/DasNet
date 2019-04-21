# DasNet
DasNet is a new multi-task learning framework to accelerate inference speed.
In this project we provide DasNet framework and apply this method on ImageNet and Birds200. Dataset also includes MIT67 and CatVsDog. 
To demonstrate the accuracy of our approach, we compared the recent popular pruning methods to a baseline. Specifically, the comparison 
algorithm has Taylor prune(Pruning convolutionalneuralnetworksforresourceefﬁcienttransferlearning), ThiNet( Thinet: A ﬁlter level pruning method for deep neural network compres
sion) and AutoPruner(Autopruner: An end-to-end trainable ﬁlter pruning method for efﬁcient deep model inference). Table 1 and Figure 1 shows our results.
![figure1]()

From the Table 1 we can see that the experimental results of the all methods are similar,but the DasNet is slightly better. In addition to inference parameter efﬁciency, another direct evaluation criterion is the amount of ﬂoating-point operations, which is a popular metric to evaluate the complexity of CNN models. Figure 2 shows the ﬂoating-point calculations of DasNet and other 3 pruning methods while using different sized network. Dataset used is CatVsDog. Our method improves floating-point operations better.


Although DasNet is mainly implemented on resnext and the structure unit of the search is at the path level, we have extended this method in order to demonstrate the versatility of our algorithm. In the common network structure optimization phase, the search structure is sequentially followed by layer, path, and filter. The optimization strategy is the same as the original DasNet. For example, for layer optimization, we can set the connection coefficient and train on the path of the layer, as shown in Figure 2. By using this method, we can apply DasNet to ResNet, ShuffleNet, DenseNet and MobileNetV2. Preliminary results are shown in the DasNet(New) line in Table 2.
