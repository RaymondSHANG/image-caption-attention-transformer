import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from models_6 import Encoder, DecoderWithTransformer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
# folder with data files saved by create_input_files.py
data_folder = '/home/yshang/Documents/imgcap2'  # '/media/ssd/caption data'
checkpoint_folder = '/home/yshang/Documents/imgcap2/checkpoints'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# model_name = 'model1'  # model1:CNN, LSTM+attension
# model_name = 'model2'  # model1:CNN, LSTM
# model_name = 'model3'  # model1:CNN, Transformer_simple
# model_name = 'model4'  # model1:CNN, Transformer_complex

model_name = 'model_6'  # model1:CNN, Transformer_full

train_logfile = model_name+".log"
# Model parameters
emb_dim = 512  # dimension of word embeddings
# attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 35
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 128
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 400  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
# checkpoint = None
# path to checkpoint, None if none
# '/home/yshang/Documents/imgcap/checkpoints/legacy_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'checkpoint_model1_loadPre_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'BEST_checkpoint_model5_coco_5_cap_per_img_5_min_word_freq.pth.tar'
checkpoint = None #'BEST_checkpoint_model5_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'BEST_checkpoint_model3_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'BEST_checkpoint_model2_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'checkpoint_model1_loadPre_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# 'checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# '/home/yshang/Documents/imgcap/checkpoints/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'


def alpha_reg(attns: Tensor, alpha_c) -> Tensor:

    # Doubly stochastic attention regularization:
    # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
    # change atten size to be
    # [layer_num, head_num, batch_size, max_len, encode_size^2]
    attns = attns.permute(0, 2, 1, 3, 4)
    ln, hn = attns.size()[:2]  # number of layers, number of heads

    # calc λ(1-∑αi)^2 for each pixel in each head in each layer
    # alphas [layer_num, head_num, batch_size*encode_size^2]
    # TODO:
    # Reduction: Would it make any difference if I sum across
    # (encode_size^2, and head) dimensions and average across batch and
    # layers?
    alphas = alpha_c * (1. - attns.sum(dim=3).view(ln, hn, -1))**2
    # alphas: Tensor
    dsar = alphas.mean(-1).sum()

    return dsar


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    pad_id = word_map["<pad>"]
    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = Encoder(encoded_image_size=14)  # , embed_dim=emb_dim
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        # Deocder
        #     "d_model": 512,
        #    "enc_ff_dim": 512,
        #    "dec_ff_dim": 2048,
        #    "enc_n_layers": 2,
        #    "dec_n_layers": 4,
        #    "enc_n_heads": 8,
        #    "dec_n_heads": 8,
        #    "dropout": 0.1
        decoder = DecoderWithTransformer(vocab_size=len(word_map),
                                         embed_dim=emb_dim,
                                         encoded_image_size=14,
                                         enc_feedforward_dim=512,
                                         dec_feedforward_dim=2048,
                                         num_dec_layer=2,
                                         num_heads_encoder=2,
                                         num_heads_decoder=2,
                                         max_len=52,  # 52-1
                                         enc_dropout=0.2,
                                         dec_dropout=0.5,
                                         pad_id=pad_id)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

    else:
        checkpoint = torch.load(checkpoint_folder+"/"+checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

        del checkpoint
        if encoder_optimizer is not None:
            update_learning_rate(encoder_optimizer, encoder_lr)
        update_learning_rate(decoder_optimizer, decoder_lr)
        print(f"loading from epoch {start_epoch-1}")
        print(f"bleu4:{best_bleu4}")
        torch.cuda.empty_cache()

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    torch.cuda.empty_cache()
    # Loss function
    # word_map['<start>'], word_map['<end>'], word_map['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=word_map['<pad>']).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer)
    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        trainloss, trainacc = train(train_loader=train_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    encoder_optimizer=encoder_optimizer,
                                    decoder_optimizer=decoder_optimizer,
                                    epoch=epoch)
        scheduler.step(trainloss)
        # One epoch's validation
        recent_bleu4, evalloss, evalacc = validate(val_loader=val_loader,
                                                   encoder=encoder,
                                                   decoder=decoder,
                                                   criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, checkpoint_folder, model_name)
        print(f"Summary for epoch {epoch}")
        print(f"Training: loss={trainloss},top5acc={trainacc}")
        print(
            f"evaluation: loss={evalloss},top5acc={evalacc},bleu4={recent_bleu4}")

        # Append to train_logfile
        file1 = open(checkpoint_folder+'/'+train_logfile, "a")  # append mode
        file1.write(
            f"{epoch}\t {trainloss}\t {trainacc}\t {evalloss}\t {evalacc}\t {recent_bleu4}\n")
        file1.close()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens)
        del imgs, caps, caplens, sort_ind
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, *_ = pack_padded_sequence(  #
            scores, decode_lengths, batch_first=True)
        targets, *_ = pack_padded_sequence(  # , _
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        # alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        alpha_trans_c = alpha_c / (2 * 2)
        loss += alpha_reg(alphas, alpha_trans_c)
        # loss += alpha_reg(alphas, alpha_c)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return losses.avg, top5accs.avg


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)  # change
            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens)

            sort_ind = sort_ind.to(device)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, *_ = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets, *_ = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            #
            # loss += alpha_reg(alphas, alpha_c)
            alpha_trans_c = alpha_c / (2 * 2)
            loss += alpha_reg(alphas, alpha_trans_c)
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, losses.avg, top5accs.avg


if __name__ == '__main__':
    main()
