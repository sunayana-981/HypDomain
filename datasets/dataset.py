import glob
import cv2
import os
import glob
from transformers import AutoProcessor

class PACS(object):
    def __init__(self,data_dir,split='train',extension="jpg"):
        self.data_dir = data_dir
        self.split = split
        self.dataset = glob.glob(os.path.join(self.data_dir,'**',f'*.{extension}'),recursive = True)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if self.split == 'train':
            self.dataset = self.dataset[:3*len(self.dataset)//4]
        elif self.split == 'test':
            self.dataset = self.dataset[3*len(self.dataset)//4:]
        

        
        domains = os.listdir(self.data_dir)
        self.domain_index = {value: index for index, value in enumerate(domains)}
        
        classes = os.listdir(os.path.join(self.data_dir,domains[0]))
        self.class_index = {value: index for index, value in enumerate(classes)}

        print(self.class_index)
        print(self.domain_index)
        self.num_classes = len(classes)
        self.num_domains = len(domains)
        
        
    def __getitem__(self, index):
        img_path = self.dataset[index]
        class_name = img_path.split("/")[-2]
        domain_name = img_path.split("/")[-3]
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        return image,self.class_index[class_name], self.domain_index[domain_name]
        
    def __len__(self):
        return len(self.dataset)

 
data = PACS(data_dir = "/ssd_scratch/cvit/souvik/PACS")
x,y,z = data[1000]
print(x.shape,y,z)