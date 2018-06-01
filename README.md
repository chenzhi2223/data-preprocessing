Here some examples using torchvision in pytroch for data preprocessing and training is presented, such as for cifar, mnist, as well as svhn. Feel free to use them for your academic or project usage. 

1. Some preprocessing techniques are implemented, e.g., ZCA whitening, which is proved to improve the accuracy in classification of such datasets. Please see the cifar.py file.

2. In addition, we provide a training example for mnist/cifar training, while we implement some functions to compute the mean and std for batchNorm layer final inference stage usage following the paper invented batch normalization. It must be noted that official pytorch uses running-average for both mean and std of different feature maps of different layers. This also works well in practice as long as you run your training hundreads of epochs. However, if you only run for several epochs, we recommend our method which strictly follows the procedure computing the global mean and std of all batches for inference stage. Please see the details in the mnist_torch.py file.  

