from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image

class iCIFAR100(CIFAR100):
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.transform=transform
        self.target_transform=target_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.train = train

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.test_data[np.array(self.test_labels) == label]
            data = np.transpose(data, (0, 2, 3, 1))
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas,labels=self.concatenate(datas,labels)
        self.TestData=datas if not len(self.TestData) else np.concatenate((self.TestData,datas),axis=0)
        self.TestLabels=labels if not len(self.TestLabels) else np.concatenate((self.TestLabels,labels),axis=0)
        
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    def getTrainData(self,classes,exemplar_set):

        datas,labels=[],[]
        if len(exemplar_set)!=0:
            datas=[exemplar for exemplar in exemplar_set ]
            length=len(datas[0])
            labels=[np.full((length),label) for label in range(len(exemplar_set))]

        for label in range(classes[0],classes[1]):
            data=self.train_data[np.array(self.train_labels)==label]
            data = np.transpose(data, (0, 2, 3, 1))
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)

        print("the size of train set is %s"%(str(self.TrainData.shape)))
        print("the size of train label is %s"%str(self.TrainLabels.shape))


    def __getitem__(self, index):
        if self.train:
            img  = Image.fromarray(self.TrainData[index]) 
            target = self.TrainLabels[index]
        else:
            img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        
        return index, img, target


    def __len__(self):
        if self.train:
            return len(self.TrainData)
        else:
            return len(self.TestData)

    def get_image_class(self,label):
        res = self.train_data[np.array(self.train_labels)==label]
        res = np.transpose(res, (0, 2, 3, 1))
        return res

