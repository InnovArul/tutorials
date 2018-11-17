Transfer learning runs:

Model: ResNet18
Dataset : CIFAR10

1) 

Pretrained: Yes (only optimize fc layer)
Data Augmentation: No (just resize to 224x224)
Hyper parameters: lr=0.0001, momentum=0.9, weight-decay=5e-4, batchsize-train=32

Performance: 

epoch-1 = 71% 
epoch-5 = 77% 
epoch-10 = 78% 

2) 

Pretrained: Yes (only optimize fc layer)
Data Augmentation: No (just resize to 224x224)
Hyper parameters: lr=0.01, momentum=0.9, weight-decay=5e-4, batchsize-train=32

Performance: 

epoch-1 = 73% 
epoch-5 = 72% 

oscillates at ~72% mark due to higher learing rate.

3) 

Pretrained: Yes (all layers are optimized)
Data Augmentation: No (just resize to 224x224)
Hyper parameters: lr=0.0001, momentum=0.9, weight-decay=5e-4, batchsize-train=32

Performance: 

epoch-1 = 69% 
epoch-2 = 80% 
epoch-3 = 84% 
epoch-4 = 86% 
epoch-5 = 88%
epoch-6 = 89% 
epoch-7 = 90% 
epoch-8 = 90% 
epoch-9 = 90% 
epoch-10 = 91% 

4) 

Pretrained: No
Data Augmentation: No (just resize to 224x224)
Hyper parameters: lr=0.01, momentum=0.9, weight-decay=5e-4, batchsize-train=32

Performance: 

epoch-1 = 56% 
epoch-2 = 70% 
epoch-3 = 73% 
epoch-4 = 73% 
epoch-5 = 74% 
epoch-6 = 75%
epoch-10 = 79% 


