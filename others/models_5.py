import math
from copy import deepcopy
from typing import Tuple
import torch
from torch import nn
import torchvision
from torch import nn, Tensor
from torch.nn import MultiheadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODED_IMG_SIZE = 14


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class transformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(transformerDecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int

        num_heads:  number of heads in the multiheadattention model.
                    int

        dropout:    dropout value
                    float
        """

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor, average_attn_weights=False) -> Tuple[Tensor, Tensor]:
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]

        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]

        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]

        tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
                        [batch_size , max_len]

        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]

        attn:           Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        To be able to do so, I have changed the code at
                        /.virtualenvs/<env_name>/lib/python3.8/site-packages/torch/nn/functional.py
                        line 4818 and changed
                        `return attn_output, attn_output_weights.sum(dim=1) /
                        num_heads` to be
                        `return attn_output, attn_output_weights`

        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        # average_attn_weights=False for multiheadAtt
        output2, attns = self.multihead_attn(
            output, enc_outputs, enc_outputs, average_attn_weights=average_attn_weights)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


class Encoder(nn.Module):
    """
    Encoder, CNN encoder
    """

    def __init__(self, encoded_image_size=ENCODED_IMG_SIZE, embed_dim=512):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # resnet = torchvision.models.resnet101(
        #    pretrained=True)  # pretrained ImageNet ResNet-101
        # weights=ResNet101_Weights.DEFAULT
        resnet = torchvision.models.resnet101(
            weights='ResNet101_Weights.DEFAULT')
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Batchnormalization
        if False:
            self.downsampling = nn.Conv2d(in_channels=2048,
                                          out_channels=embed_dim,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False)
            self.bn = nn.BatchNorm2d(embed_dim)
            self.relu = nn.ReLU(inplace=True)

        # Resize images, use 2D adaptive max pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(encoded_image_size)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        out = self.resnet(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        # out = self.relu(self.bn(self.downsampling(out)))

        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        out = self.adaptive_pool(out)
        # out = self.batchnorm(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:  # was 5: previously
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderWithTransformer(nn.Module):
    """
    Decoder with transformer
    The input is (N,W,H,embed_dim), which will firstly to (N,W*H,embed_dim) as input for the whole transformer
    The transformer itself will contain both encoder and decoder
    The transformer encoder will include MultiheadAttention layer followed by LayerNorm, then FeedForwardLayer
    The transformer decoder will include 
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 encoded_image_size: int,
                 enc_feedforward_dim: int = 512,
                 dec_feedforward_dim: int = 2048,
                 num_dec_layer: int = 2,
                 num_heads_encoder: int = 2,
                 num_heads_decoder: int = 2,
                 max_len: int = 52,  # 52-1
                 enc_dropout: float = 0.1,
                 dec_dropout: float = 0.5,
                 pad_id: int = 0
                 ):
        """
        :param embed_dim: embedding size, which is also hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param decoder_dim: size of decoder's RNN, to be simple, be the same as enbeded_dim,d_model
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :encoder_numpixel=196,this number should be the W*H value from encoder outputs
        """
        super(DecoderWithTransformer, self).__init__()
        self.pad_id = pad_id

        ###############################   TransformerEncoder, assuming only 1 encoder layer  ##################################
        # First of all, we need to change C_input to embed_dim
        self.enc_downsampling = nn.Conv2d(in_channels=2048,
                                          out_channels=embed_dim,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False)
        self.enc_embed_bn = nn.BatchNorm2d(embed_dim)
        self.enc_embed_relu = nn.ReLU(inplace=True)

        self.enc_dropout = nn.Dropout(p=enc_dropout)

        ###############################   TransformerDecoder  with num_dec_layer layers  ##################################
        # Embedding layer + pos encoding
        decoder_layer = transformerDecoderLayer(d_model=embed_dim,
                                                num_heads=num_heads_decoder,
                                                feedforward_dim=dec_feedforward_dim,
                                                dropout=dec_dropout)

        self.cptn_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(embed_dim, max_len)
        self.decodelayers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_dec_layer)])

        self.dropout = nn.Dropout(p=dec_dropout)

        ###############################   Final Layer  ##################################
        self.predictor = nn.Linear(embed_dim, vocab_size, bias=False)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)

        # image embedding
        # First change (N,W,H,C) -> (N,C,W,H) ->conv ->(N,embed_dim,W,H) ->(N,W,H,embed_dim)
        encoder_out = self.enc_embed_relu(
            self.enc_embed_bn(self.enc_downsampling(encoder_out.permute(0, 3, 1, 2))))
        encoder_out = encoder_out.permute(0, 2, 3, 1)
        encoder_dim = encoder_out.size(-1)  # img_embed_dim
        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        ###################################    Tansformer encoder    ##########################
        # transformer_encoder forward
        # self-attention
        # [encode_size^2, batch_size, embed_dim]
        enc_output = encoder_out.permute(1, 0, 2)

        # after encoder, we have ([encode_size^2, batch_size, embed_dim])
        ###################################    Tansformer decoder    ##########################

        # dec_inputs:     Captions to decode [max_len, batch_size, embed_dim]

        # enc_output:    Encoded image to decode [encode_size^2=196, batch_size, embed_dim]

        # tgt_mask:       Mask to ensure that decoder doesn't look at future
        #                 tokens from a given subsequence
        #                 [max_len , max_len]

        # tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
        #                [batch_size , max_len]
        # self attention + resedual summation + norm

        # create masks, then pass to decoder
        tgt_pad_mask = (encoded_captions == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(encoded_captions.size()[1])
        tgt_mask = tgt_mask.to(device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(encoded_captions)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.decodelayers:
            tgt_cptn, attns = layer(
                tgt_cptn, enc_output, tgt_mask, tgt_pad_mask, average_attn_weights=False)
            # Attension weights,[layer_num, batch_size, head_num, max_len,encode_size^2]
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        # return tgt_cptn, attns_all
        # attns.contiguous(),
        ###################################   Final Layer ################################

        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # type: Tensor

        return predictions.contiguous(), encoded_captions, decode_lengths, attns_all.contiguous(), sort_ind
