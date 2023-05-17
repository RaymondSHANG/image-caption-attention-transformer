import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkdir = '/home/yshang/Documents/imgcap2/checkpoints/'
bestmodel2 = 'checkpoint_model_trans5_coco_5_cap_per_img_5_min_word_freq.pth.tar'
bestmodel3 = 'BEST_checkpoint_model_trans5_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# '/home/yshang/Documents/imgcap/checkpoints/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model_path = checkdir + \
    bestmodel2  # test.pth.tar
checkpoint = torch.load(model_path, map_location=str(device))
print(f"epoch:{checkpoint['epoch']}")
print(f"epoch:{checkpoint['epochs_since_improvement']}")
print(f"blue4:{checkpoint['bleu-4']}")
# print(f"encoder structure: {checkpoint['encoder']}")
print("\n\n##################################\n")
print("Model Best\n")
model_path = checkdir + \
    bestmodel3  # test.pth.tar
checkpoint = torch.load(model_path, map_location=str(device))
print(f"epoch:{checkpoint['epoch']}")
print(f"blue4:{checkpoint['bleu-4']}")
