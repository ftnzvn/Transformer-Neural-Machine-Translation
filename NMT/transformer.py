#   Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017) <https://arxiv.org/abs/1706.03762>`.
#   This project was part of the University of Edinburgh's Master's course NLU+.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models.transformer_helper import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEmbedding, generate_embedding, fill_with_neg_inf

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    """
    Transformer Model Class. Inherits from Seq2SeqModel and calls TransformerEncoder and TransformerDecoder submodels.
    """
    def __init__(self,
                 args,
                 encoder,
                 decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--no-scale-embedding', action='store_true', help='if True, dont scale embeddings')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Construct model. """
        base_architecture(args)

        if getattr(args, 'max_source_positions', None) is None:
            args.max_src_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_tgt_positions = DEFAULT_MAX_TARGET_POSITIONS

        # Transformer Encoder
        encoder = TransformerEncoder(args, src_dict)
        decoder = TransformerDecoder(args, tgt_dict)
        return cls(args, encoder, decoder)


class TransformerEncoder(Seq2SeqEncoder):
    """ Defines an encoder class. """

    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_src_positions = args.max_src_positions
        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_src_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Encoder Layers
        self.layers.extend([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # Embed tokens indices
        embeddings = self.embed_scale * self.embedding(src_tokens)

        # Clone for output state
        src_embeddings = embeddings.clone()

        '''
        ___QUESTION-6-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  What is the purpose of the positional embeddings in the encoder and decoder? 
        Positional embeddings provide the encoder and decoder with a sense of the order of the sequence, it encodes the position of each token.
        The transformer model processes input sequences in parallel, it does not account for the sequential nature of the data in and of itself.
        The positional embeddings are needed, otherwise the self-attention mechanism would be permutation-equivariant.
        This, in turn, allows the model to better understand the input sequence and generate a better quality and more coherent output.
        '''
        embeddings += self.embed_positions(src_tokens)    # embeddings.size = [batch_size, src_time_steps, embed_dim]
        '''
        ___QUESTION-6-DESCRIBE-A-END___
        '''
        forward_state = F.dropout(embeddings, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Compute padding mask for attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Forward pass through each Transformer Encoder Layer
        for layer in self.layers:
            forward_state = layer(state=forward_state, encoder_padding_mask=encoder_padding_mask)

        return {
            "src_out": forward_state,   # [src_time_steps, batch_size, num_features]
            "src_embeddings": src_embeddings,   # [batch_size, src_time_steps, num_features]
            "src_padding_mask": encoder_padding_mask,   # [batch_size, src_time_steps]
            "src_states": []    # List[]
        }


class TransformerDecoder(Seq2SeqDecoder):
    """ Defines an decoder class. """
    def __init__(self,
                 args,
                 dictionary):

        super().__init__(dictionary)
        self.dropout = args.dropout
        self.embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        self.padding_idx = dictionary.pad_idx
        self.max_tgt_positions = args.max_tgt_positions

        self.embedding = generate_embedding(len(dictionary), self.embed_dim, dictionary.pad_idx)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        self.embed_positions = PositionalEmbedding(
            self.embed_dim, padding_idx=self.padding_idx, init_size=self.max_tgt_positions + self.padding_idx + 1
        )

        self.layers = nn.ModuleList([])

        # Generate N identical Decoder Layers
        self.layers.extend([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])

        self.embed_out = nn.Linear(self.output_embed_dim, len(dictionary))

        nn.init.normal_(self.embed_out.weight, mean=0, std=self.output_embed_dim ** -0.5)

    def forward(self, tgt_inputs, encoder_out=None, incremental_state=None, features_only=False):
        # Embed positions
        positions = self.embed_positions(tgt_inputs, incremental_state=incremental_state)

        # Incremental decoding only needs the single previous token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        forward_state = self.embed_scale * self.embedding(tgt_inputs)
        forward_state += positions
        forward_state = F.dropout(forward_state, p=self.dropout, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        forward_state = forward_state.transpose(0, 1)

        # Generate padding mask
        self_attn_padding_mask = tgt_inputs.eq(self.padding_idx) if tgt_inputs.eq(self.padding_idx).any() else None

        # Forward pass through each Transformer Decode Layer
        attn_state = None
        inner_states = [forward_state]

        for layer_idx, layer in enumerate(self.layers):
            is_attention_layer = layer_idx == len(self.layers) - 1
            encoder_state = encoder_out['src_out'] if encoder_out is not None else None
            '''
            ___QUESTION-6-DESCRIBE-B-START___
            1.  Add tensor shape annotation to each of the output tensor
            
            2.  What is the purpose of self_attn_mask? 
            The purpose of self_attn_mask is to prevent information leakage by masking future tokens in a sequence during decoding.
            This ensures that the decoder is autoregressive, i.e., that the prediction at each position is only influenced by previous positions.
            In self-attention, each token in the sequence can attend to all other tokens in the same sequence, but it should not have access to future tokens in the sequence.
            The self_attn_mask applies a mask of a very large negative number on future tokens, making the softmax output for those positions 0, ensuring they have no influence
            over the generation of current tokens.
            
            3.  Why do we need it in the decoder but not in the encoder?
            Because the encoder is not generating, while the decoder is.
            The encoder is not predicting a sequence, so there is no need to block future information which affects prediction. 
            The decoder, on the other hand, is generating a sequence by predicting tokens, so we need to make sure it is not looking
            at future tokens which might bias the predictions.
            
            4.  Why do we not need a mask for incremental decoding?
            Because incremental decoding predicts the output one token at a time, using the information it has accumulated from previously generated tokens and the context provided by the encoder's latent vector.
            For each new token it generates, it only considers the previous tokens, it cannot consider future tokens.
            So there is no need for a mask to prevent looking ahead.
            '''
            self_attn_mask = self.buffered_future_mask(forward_state) if incremental_state is None else None   # self_attn_mask.size = [tgt_time_steps, src_time_steps]
            '''
            ___QUESTION-6-DESCRIBE-B-END___
            '''

            forward_state, layer_attn = layer(state=forward_state,
                                              encoder_out=encoder_state,
                                              self_attn_mask=self_attn_mask,
                                              self_attn_padding_mask=self_attn_padding_mask,
                                              need_attn=is_attention_layer,
                                              need_head_weights=is_attention_layer)
            inner_states.append(forward_state)

            if layer_attn is not None and is_attention_layer:
                attn_state = layer_attn.float()

        if attn_state is not None:
            attn_state = attn_state.mean(dim=0)

        forward_state = forward_state.transpose(0, 1)

        # Project into output layer
        if not features_only:
            '''
            ___QUESTION-6-DESCRIBE-C-START___
            1.  Why do we need a linear projection after the decoder layers? 
            To transform the output feature dimension into the vocabulary size.
            The linear projection results in a vector with scores for each item in the vocabulary.
            We then apply a softmax function to these scores, which gives us a probability
            distribution over the words in the target vocabulary.

            2.  What would the output represent if features_only=True?
            If features_only=True, the model would skip the final linear projection layer which maps decoder output features to the vocabulary size
            The output would consist of the raw feature representations at each timestep.
            These represent the high-dimensional embeddings of potential output tokens before being translated into actual token predictions.
            '''
            forward_state = self.embed_out(forward_state)
            '''
            ___QUESTION-6-DESCRIBE-C-END___
            '''
        return forward_state, {
            "attn_state": attn_state,
            "inner_states": inner_states
        }

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (not hasattr(self, '_future_mask')) or self._future_mask is None or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    # Enc-Dec params [mostly tied for simplicity]
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', args.encoder_layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)

    # Dropout and activation
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.2)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
