import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
# import cv2  # cv2.resize
# import imageio  # imageio.imread
# from scipy.misc import imageio.imread, cv2.imread
# from skimage.io import imread
# from skimage.transform import resize as imresize
from cv2 import imread
from cv2 import resize as imresize  # cv2.resize
from PIL import Image
# from attrdict import AttrDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "model_trans5"


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    Caption_End = False
    vocab_size = len(word_map)
    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
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
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']] * 52] * k).to(device)  # (k, 52)

    # Tensor to store top k sequences; now they're just <start>
    seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size1, enc_image_size2).to(
        device)  # (k, 1, enc_image_size, enc_image_size)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
        scores, _, _, alpha_dict, _ = decoder(
            encoder_out, k_prev_words, cap_len)
        # [s, 1, vocab_size] -> [s, vocab_size]
        scores = scores[:, step - 1, :].squeeze(1)
        # choose the last layer, transformer decoder is comosed of a stack of 2 identical layers.
        # [s, n_heads=2, len_q=52, len_k=196]
        # print(f"alpha_dict dimenstion{alpha_dict.size()}")
        alpha = alpha_dict[-1]
        # print(f"alpha dimenstion{alpha.size()}")
        # TODO: AVG Attention to Visualize
        # for i in range(len(alpha_dict["dec_enc_attns"])):
        #     n_heads = alpha_dict["dec_enc_attns"][i].size(1)
        #     for j in range(n_heads):
        #         pass
        # the second dim corresponds to the Multi-head attention = 8, now 0
        # the third dim corresponds to cur caption position
        # [s, 1, enc_image_size, enc_image_size]
        alpha = alpha[:, 0, step-1,
                      :].view(k, 1, enc_image_size1, enc_image_size2)

        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            # (s)
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        # Add new words to sequences, alphas
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        # (s, step+1, enc_image_size, enc_image_size)
        seqs_alpha = torch.cat(
            [seqs_alpha[prev_word_inds], alpha[prev_word_inds]], dim=1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))
        # Set aside complete sequences
        if len(complete_inds) > 0:
            Caption_End = True
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        k_prev_words = k_prev_words[incomplete_inds]
        k_prev_words[:, :step + 1] = seqs  # [s, 52]
        # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    assert Caption_End
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black',
                 backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    # plt.show()
    image_pathNoext = image_path.rsplit('.', maxsplit=1)[0]
    plt.savefig(image_pathNoext+"_"+model_name+"_cap.jpeg")


class Bar(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5,
                        type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')

    # args = parser.parse_args()
    # args = AttrDict()
    args = Bar({"img": 'myimg/img1.jpeg',
                "model": '/home/yshang/Documents/imgcap2/checkpoints/BEST_checkpoint_model_trans5_coco_5_cap_per_img_5_min_word_freq.pth.tar',
                "word_map": '/home/yshang/Documents/imgcap2/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                "beam_size": 5,
                "smooth": False
                })
    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
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
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(
        encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)

# python caption.py --img='myimg/img1.jpeg' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load('checkpoints/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', map_location=str(device))
