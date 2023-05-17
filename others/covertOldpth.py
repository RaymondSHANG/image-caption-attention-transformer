import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = 'checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
checkpoint = torch.load(model_path, map_location=str(device))
#model = torch.jit.load(model_path, 'cpu')
#model = model.state_dict()
test1 = checkpoint['encoder_optimizer'].state_dict() #Pass
test2 = checkpoint['decoder_optimizer'].state_dict() #Pass
test3 = checkpoint['decoder'].state_dict() #Pass
test4 = checkpoint['encoder'].state_dict() #Pass

state = {'epoch': checkpoint['epoch'],
             'epochs_since_improvement': checkpoint['epochs_since_improvement'],
             'bleu-4': checkpoint['bleu-4'],
             'encoder': test4,
             'decoder': test3,
             'encoder_optimizer': test1,
             'decoder_optimizer': test2}

torch.save(state, 'test2.pth.tar')


#torch.save(model, 'your_path.pth.tar')

#####another solution
checkpoint = torch.load(checkpoint)
torch.save(checkpoint, 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')

#Save and recall model to resolve

#code : model dim size = 512
#pretraning model dim size = 2048
#####


# dict elements
# epoch
# epochs_since_improvement
# bleu-4
# encoder
# decoder
# encoder_optimizer
# decoder_optimizer


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['encoder_optimizer'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)