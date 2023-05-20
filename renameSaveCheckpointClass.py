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
    encoder.fine_tune(True)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                        lr=args.encoder_lr)
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
                                        num_dec_layer=2,
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
                                        num_dec_layer=1,
                                        num_heads_encoder=2,
                                        num_heads_decoder=2,
                                        max_len=52,  # 52-1
                                        enc_dropout=0.2,
                                        dec_dropout=0.5,
                                        pad_id=pad_id)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                lr=args.decoder_lr)  


    checkpoint = torch.load(args.checkpoint_folder+"/"+args.checkpoint)
    #encoder.load_state_dict(checkpoint['modelA_state_dict'])
    #decoder.load_state_dict(checkpoint['modelB_state_dict'])
    #encoder_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
    #decoder_optimizer.load_state_dict(checkpoint['optimizerB_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']

    del checkpoint
    print(f"loading from epoch {start_epoch-1}")
    print(f"bleu4:{best_bleu4}")
    torch.cuda.empty_cache()

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    
    save_checkpoint(args.data_name, start_epoch-1, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, best_bleu4, True, args.checkpoint_folder, args.model_name)
    print(f"resaveed for epoch {start_epoch-1}")


if __name__ == '__main__':
    main()
