import numpy as np
import torch

def calculat_index(mylist):
    anomly_class_num= [1,36,3,2,7,5]
    #anomly_class_num= [1]

    my_indecies=[]
    class_num=[]


    for a in range(len(mylist)):

        if mylist[a] in anomly_class_num:
            my_indecies.append(a)
            class_num.append(mylist[a])

    return my_indecies , class_num

def claculate_mask(pred_mask):

    my_mask = torch.zeros(pred_mask.shape[1],pred_mask.shape[2])

    for i in range(pred_mask.shape[0]):
        my_mask[pred_mask[i] == True] = 255



    return my_mask.cpu().detach().numpy()

