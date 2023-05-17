import yaml
import argparse
import os
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
parser = argparse.ArgumentParser(description='Image caption model settings')
parser.add_argument('--config', default='./configs/config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f,Loader=yaml.Loader)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)

# folder with data files saved by create_input_files.py
data_folder = args.data_folder  # 'caption data'
data_name = args.data_name  # base name shared by data files
checkfilename = 'checkpoint_' + args.model_name+'_' +\
        data_name + '.pth.tar'
match args.checkpoint_eval:
    case 'Best':
        checkfilename = 'BEST_checkpoint_' + args.model_name+'_' +\
                        data_name + '.pth.tar'
    case 'Last':
        checkfilename = 'checkpoint_' + args.model_name+'_' +\
                        data_name + '.pth.tar'
#os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')

# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(os.path.join(args.checkpoint_folder,checkfilename))
print(f"loading from epoch {checkpoint['epoch']}")
print(f"bleu4:{checkpoint['bleu-4']}")
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
del checkpoint
torch.cuda.empty_cache()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate_LSTM(Attention,beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)
            if Attention:
                # (s, encoder_dim), (s, num_pixels)
                awe, _ = decoder.attention(encoder_out, h)

                # gating scalar, (s, encoder_dim)
                gate = decoder.sigmoid(decoder.f_beta(h))
                awe = gate * awe
            else:
                gate = decoder.sigmoid(decoder.f_beta(h))
                awe = gate  # * awe
                alpha = torch.ones(
                    (awe.size(dim=0), enc_image_size, enc_image_size)).to(device)
                alpha = alpha*1.0/(enc_image_size*enc_image_size)
            
            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1) #.long()

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds].long()]  # long()
            c = c[prev_word_inds[incomplete_inds].long()]  # long()
            # long()
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {
                          word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4

def evaluate_transformer(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(image)
        enc_image_size1 = encoder_out.size(1)
        enc_image_size2 = encoder_out.size(2)
        encoder_dim = encoder_out.size(3)

        # We'll treat the problem as having a batch size of k
        # (k, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder_out.expand(
            k, enc_image_size1, enc_image_size2, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        # k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        # Important: [1, 52] (eg: [[<start> <start> <start> ...]]) will not work, since it contains the position encoding
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]*52] * k).to(device)  # (k, 52)

        # Tensor to store top k sequences; now they're just <start>
        # seqs = k_prev_words  # (k, 1)
        seqs = torch.LongTensor([[word_map['<start>']]]
                                * k).to(device)  # (k, 1)
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            # print("steps {} k_prev_words: {}".format(step, k_prev_words))
            # cap_len = torch.LongTensor([52]).repeat(k, 1).to(device) may cause different sorted results on GPU/CPU in transformer.py
            cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
            # scores, _, _, _, _ = decoder(encoder_out, k_prev_words, cap_len)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                encoder_out, k_prev_words, cap_len)

            # [s, 1, vocab_size] -> [s, vocab_size]
            scores = scores[:, step-1, :].squeeze(1)
            scores = F.log_softmax(scores, dim=1)
            # top_k_scores: [s, 1]
            scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                Caption_End = True
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            # Important: this will not work, since decoder has self-attention
            # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step+1] = seqs  # [s, 52]
            # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        # choose the caption which has the best_score.
        assert Caption_End
        indices = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[indices]
        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)
        # Hypotheses
        # tmp_hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append([w for w in seq if w not in {
            word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        assert len(references) == len(hypotheses)
        # Print References, Hypotheses and metrics every step
        # words = []
        # # print('*' * 10 + 'ImageCaptions' + '*' * 10, len(img_captions))
        # for seq in img_captions:
        #     words.append([rev_word_map[ind] for ind in seq])
        # for i, seq in enumerate(words):
        #     print('Reference{}: '.format(i), seq)
        # print('Hypotheses: ', [rev_word_map[ind] for ind in tmp_hyp])
        # metrics = get_eval_score([img_captions], [tmp_hyp])
        # print("{} - beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
        #       (args.decoder_mode, args.beam_size, metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"],
        #        metrics["Bleu_4"],
        #        metrics["METEOR"], metrics["ROUGE_L"], metrics["CIDEr"]))

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    # metrics = get_eval_score(references, hypotheses)
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4

if __name__ == '__main__':
    beam_size = args.beam_size_eval
    match args.model_name:
        case 'model_1'|'model_2'|'model_3':
            evalresult = evaluate_LSTM(True,beam_size)
        case 'model_3':
            evalresult = evaluate_LSTM(False,beam_size)
        case 'model_4'|'model_5'|'model_6'|'model_7':
            evalresult = evaluate_transformer(beam_size)

    print("\nBLEU-4 score @ beam size of %d is %.4f." %
          (beam_size, evalresult))
