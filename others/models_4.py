import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODED_IMG_SIZE = 14


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=ENCODED_IMG_SIZE):
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

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))
        # self.batchnorm = nn.BatchNorm2d(
        #    (encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
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
    Decoder.
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, num_heads=2, dim_feedforward=2048, encoder_imgwidth=ENCODED_IMG_SIZE, encoder_dim=2048, dropout=0.5, dim_k=96, dim_v=96, dim_q=96):
        """
        :param embed_dim: embedding size, which is also hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param decoder_dim: size of decoder's RNN, to be simple, be the same as enbeded_dim
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :encoder_numpixel=196,this number should be the W*H value from encoder outputs
        """
        super(DecoderWithTransformer, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.decoder_dim = decoder_dim  # decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.encoder_size = encoder_imgwidth*encoder_imgwidth
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads

        # change encoder channels to embed_dim/hidden_dim
        self.encoder_enbedding = nn.Linear(encoder_dim, embed_dim)
        self.posembeddingL = nn.Embedding(
            num_embeddings=self.encoder_size, embedding_dim=self.embed_dim)

        # 2 MHA
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)

        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(
            self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        #
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.dropoutmh = nn.Dropout(p=0.2)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.dim_feedforward, self.hidden_dim),
        )
        self.norm_feedforward = nn.LayerNorm(self.hidden_dim)
        self.dropoutff = nn.Dropout(p=0.2)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final = nn.Linear(self.hidden_dim, vocab_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def embed(self, encoder_out):
        embeddings = None
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        # inputs = inputs.to(self.device)
        embeddings_word = self.encoder_enbedding(encoder_out)

        word_pos = torch.Tensor(range(encoder_out.size(1))).to(
            torch.long).to(device)
        embeddings_pos = self.posembeddingL(word_pos)

        embeddings = (embeddings_word + embeddings_pos)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)

        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """

        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        outputs = None
        # inputs = inputs.to(self.device)
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        dot_p1 = torch.bmm(q1, torch.transpose(k1, 1, 2))
        d1 = self.dim_k ** 0.5
        sf1 = self.softmax(dot_p1/d1)
        head1 = torch.bmm(sf1, v1)

        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)
        dot_p2 = torch.bmm(q2, torch.transpose(k2, 1, 2))
        d2 = self.dim_k ** 0.5
        sf2 = self.softmax(dot_p2/d2)
        head2 = torch.bmm(sf2, v2)

        head_cat = torch.cat((head1, head2), 2)
        outputs = self.attention_head_projection(head_cat)
        outputs = inputs + outputs
        outputs = self.norm_mh(outputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """

        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = None
        # inputs = inputs.to(self.device)
        outputs = self.feedforward(inputs)
        outputs = self.norm_feedforward(inputs + outputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """

        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = None
        # inputs = inputs.to(self.device)
        outputs = self.final(inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # embedding encoder_out
        # we need positioning embedding, and a linear layer to change num_pixel to hidden_dim
        outputs = self.embed(encoder_out)
        outputs = self.multi_head_attention(outputs)
        outputs = self.dropoutmh(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.dropoutff(outputs)
        outputs = self.final_layer(outputs)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        return outputs, encoded_captions, decode_lengths, sort_ind
