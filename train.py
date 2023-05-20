import yaml
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from models.encoder import Encoder,EncoderMLPMixer
from models.decoder_LSTM import DecoderWithAttention,DecoderWithoutAttention
from models.DecoderWithTransformerSelfAtt import DecoderWithTransformerSelfAtt
from models.DecoderWithTransformerPytorch import DecoderWithTransformerCrossAtt,DecoderWithTransformerOnlyDecoder,DecoderWithTransformerFull
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu



# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True


def main():
    """
    Training ///and validation.
    """
    #global args
    parser = argparse.ArgumentParser(description='Image caption model settings')
    parser.add_argument('--config', default='./configs/config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f,Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    global word_map
    train_logfile = args.model_name+"training.log"
    start_epoch = args.start_epoch
    epochs_since_improvement=args.epochs_since_improvement
    best_bleu4=args.best_bleu4
    #grad_clip=args.grad_clip
    #print_freq=args.print_freq
    cudnn.benchmark = args.cudnn_benchmark
    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    pad_id = word_map["<pad>"]
    # Initialize / load checkpoint
    
    if args.model_name == 'model_2':
        encoder = EncoderMLPMixer()
    else:
        encoder = Encoder()
    encoder.fine_tune(args.fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr=args.encoder_lr) if args.fine_tune_encoder else None
    match args.model_name:
        case 'model_1':
            decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                        embed_dim=args.emb_dim,
                                        decoder_dim=args.decoder_dim,
                                        vocab_size=len(word_map),
                                        dropout=args.dropout)
        case 'model_2':
            decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                        embed_dim=args.emb_dim,
                                        decoder_dim=args.decoder_dim,
                                        vocab_size=len(word_map),
                                        encoder_dim=1024,
                                        dropout=args.dropout)
        case 'model_3':
            decoder = DecoderWithoutAttention(embed_dim=args.emb_dim,
                        decoder_dim=args.decoder_dim,
                        vocab_size=len(word_map),
                        dropout=args.dropout)
        case 'model_4':
            decoder = DecoderWithTransformerSelfAtt(embed_dim=args.emb_dim,
                                        decoder_dim=args.decoder_dim,
                                        vocab_size=len(word_map),
                                        dim_k=128, dim_v=128, dim_q=128,
                                        dropout=args.dropout)
        case 'model_5':
            decoder = DecoderWithTransformerCrossAtt(vocab_size=len(word_map),
                                        embed_dim=args.emb_dim,
                                        encoded_image_size=14,
                                        enc_feedforward_dim=512,
                                        dec_feedforward_dim=2048,
                                        num_dec_layer=1,
                                        num_heads_encoder=2,
                                        num_heads_decoder=2,
                                        max_len=52,  # 52-1
                                        enc_dropout=0.2,
                                        dec_dropout=0.5,
                                        pad_id=pad_id)
        case 'model_6':
            decoder = DecoderWithTransformerOnlyDecoder(vocab_size=len(word_map),
                                        embed_dim=args.emb_dim,
                                        encoded_image_size=14,
                                        enc_feedforward_dim=512,
                                        dec_feedforward_dim=2048,
                                        num_dec_layer=1,
                                        num_heads_encoder=2,
                                        num_heads_decoder=2,
                                        max_len=52,  # 52-1
                                        enc_dropout=0.2,
                                        dec_dropout=0.5,
                                        pad_id=pad_id)
        case 'model_7':
            decoder = DecoderWithTransformerFull(vocab_size=len(word_map),
                                        embed_dim=args.emb_dim,
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
                                                lr=args.decoder_lr)  

    if args.checkpoint != 'None':
        checkpoint = torch.load(args.checkpoint_folder+"/"+args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder.load_state_dict(checkpoint['decoder_state_dict']) 
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
        del checkpoint
        print(f"loading from epoch {start_epoch-1}")
        print(f"bleu4:{best_bleu4}")
        torch.cuda.empty_cache()

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    optimizer_to(encoder_optimizer,device)
    optimizer_to(decoder_optimizer,device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        trainloss, trainacc = train(args,
                                    train_loader=train_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    encoder_optimizer=encoder_optimizer,
                                    decoder_optimizer=decoder_optimizer,
                                    epoch=epoch)

        # One epoch's validation
        recent_bleu4, evalloss, evalacc = validate(args,
                                                   val_loader=val_loader,
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
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, args.checkpoint_folder, args.model_name)
        print(f"Summary for epoch {epoch}")
        print(f"Training: loss={trainloss},top5acc={trainacc}")
        print(
            f"evaluation: loss={evalloss},top5acc={evalacc},bleu4={recent_bleu4}")

        # Append to train_logfile
        file1 = open(args.checkpoint_folder+'/'+train_logfile, "a")  # append mode
        file1.write(
            f"{epoch}\t {trainloss}\t {trainacc}\t {evalloss}\t {evalacc}\t {recent_bleu4}\n")
        file1.close()

def reg_transformer(attns: Tensor, alpha_c) -> Tensor:
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

def reg_Att(attns: Tensor, alpha_c) -> Tensor:
    # Doubly stochastic attention regularization:
    # "Show, Attend and Tell" - arXiv:1502.03044v3 eq(14)
    dsar = alpha_c * ((1. - attns.sum(dim=1)) ** 2).mean()    
    return dsar
def reg_loss(args,attns: Tensor) -> Tensor: 
    if args.model_name=="model_1" or args.model_name=="model_2":
        return reg_Att(attns, args.alpha_c)
    elif args.model_name=="model_3" or args.model_name=="model_4":
        return 0
    else:
        #tansformers,only reg the crossAtt
        #reduce weight by (Nheader*NLayers)
        alpha_trans_c = args.alpha_c / (2 * 2)
        return reg_transformer(attns, alpha_trans_c)

def train(args,train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
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
        
        loss += reg_loss(args,alphas)
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

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
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return losses.avg, top5accs.avg


def validate(args,val_loader, encoder, decoder, criterion):
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
            #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss += reg_loss(args,alphas)
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq == 0:
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
