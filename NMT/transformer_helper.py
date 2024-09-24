import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  What is the purpose of encoder_padding_mask? 
        The encoder padding mask in the forward pass is used so as to ignore the padding
        tokens in the input sequence. The paddings tokens do not provide meaningful information
        and therefore do not affect the model's attention mechanisms or predictions. 
        The self-attention mechanism is applied to the input state with the query, key, and value all set to the current state.
        The encoder_padding_mask is passed as the key_padding_mask argument to the self-attention function.
        Inside the self-attention module, this mask is used to adjust the attention scores, ensuring that the model does not attend to padding tokens.
        By focusing attention only on relevant information, translation quality in NMT tasks should improve.
        '''
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  How does encoder attention differ from self attention? 
        In self-attention, the keys, values, and queries are generated from the same sequence, they are all set to the same input state. 
        In encoder attention, the queries come from the current decoder state, while the keys and values are derived from the encoder's output.
        Self-attention allows positions to attend within the same sequence, capturing internal dependencies, relations and context within that sequence.
        Encoder attention, on the other hand, allows the decoder to attend to the encoder's output, focusing on the relevant parts of the input sequence when generating the output.
        Encoder attention ensures that the output remains relevant to the input, connecting the encoder to the decoder.
        Both these attention mechanisms are crucial in machine translation, where it is important to both maintain coherence and dependencies, and to align the souce language with the target language.

        2.  What is the difference between key_padding_mask and attn_mask? 
        Although both influence attention weights and specify which keys should not be attended to by each query, they do it in different ways.
        key_padding_mask is used to prevent the model from attending to padding positions within the input sequence.
        attn_mask specifies which tokens should not be attended to, and states what key-value pairs are valid.
        key_padding_mask ignores all padding tokens in general, whereas attn_mask implements more specified attention patterns,
        which can vary for each query-key pair. attn_mask does not only ignore padding tokens, it can ignore any key based on more fine-grained criteria.
        The purpose of key_padding_mask is to ignore padding tokens across all queries, whereas the purpose of attn_mask is to control the attention mechanism in more detail, for example to enforce causal attention.
        
        3.  If you understand this difference, then why don't we need to give attn_mask here?
        This has to do with the fact that we are using encoder-attention in the following code snippet instead of self attention, as well as with the specific task of NMT.
        Encoder-attention uses the full context provided by the encoder, the decoder needs access to the entire encoder output when generating each token.
        Masking specific positions is not needed to prevent information leakage when using encoder-attention, since the encoder output represents the complete input sequence, unlike in self-attention which has sequential constraints.
        Given that our task at hand is machine translation, the encoder needs access to the entire output of the encoder to generate each token in the output sequence.
        In a task like Question Answering, it may have been more important to attend to specific parts of the encoder output, but in translation tasks the decoder might need to look at any part of the input sequence for any point in the generation process,
        especially if the translation is done on document level and not on sentence-level. 
        '''
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT
        # attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        # attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, -1)) if need_weights else None

        keys = self.k_proj(key)    # keys.size = [src_time_steps, batch_size, embed_dim]
        values = self.v_proj(value)    # values.size = [src_time_steps, batch_size, embed_dim]
        queries = self.q_proj(query)    # queries.size = [tgt_time_steps, batch_size, embed_dim]
        
        parallel_keys = keys.transpose(0, 1)    # parallel_keys.size = [batch_size, src_time_steps, embed_dim]
        # parallel_keys.size = [batch_size, src_time_steps, num_heads, head_embed_size]
        parallel_keys = parallel_keys.contiguous().view(batch_size, key.size(0), self.num_heads, self.head_embed_size)
        
        parallel_queries = queries.transpose(0, 1)    # parallel_queries.size = [batch_size, tgt_time_steps, embed_dim]
        # parallel_queries.size = [batch_size, tgt_time_steps, num_heads, head_embed_size]
        parallel_queries = parallel_queries.contiguous().view(batch_size, tgt_time_steps, self.num_heads, self.head_embed_size)
        
        parallel_values = values.transpose(0, 1)    # parallel_values.size = [batch_size, src_time_steps, embed_dim]
        # parallel_values.size = [batch_size, src_time_steps, num_heads, head_embed_size]
        parallel_values = parallel_values.contiguous().view(batch_size, value.size(0), self.num_heads, self.head_embed_size)

        parallel_keys = parallel_keys.transpose(1, 2)  # parallel_keys.size = [batch_size, num_heads, src_time_steps, head_embed_size]
        # parallel_keys.size = [batch_size * num_heads, src_time_steps, head_embed_size]
        parallel_keys = parallel_keys.contiguous().view(batch_size * self.num_heads, key.size(0), self.head_embed_size)
        
        parallel_queries = parallel_queries.transpose(1, 2)  # parallel_queries.size = [batch_size, num_heads, tgt_time_steps, head_embed_size]
        # parallel_queries.size = [batch_size * num_heads, tgt_time_steps, head_embed_size]
        parallel_queries = parallel_queries.contiguous().view(batch_size * self.num_heads, tgt_time_steps, self.head_embed_size)
        
        parallel_values = parallel_values.transpose(1, 2)  # parallel_values.size = [batch_size, num_heads, src_time_steps, head_embed_size]
        # parallel_values.size = [batch_size * num_heads, src_time_steps, head_embed_size]
        parallel_values = parallel_values.contiguous().view(batch_size * self.num_heads, value.size(0), self.head_embed_size)
        
        qk = torch.bmm(parallel_queries, parallel_keys.transpose(1, 2))  # qk.size = [batch_size * num_heads, tgt_time_steps, src_time_steps]
        
        if key_padding_mask is not None:
            qk = qk.view(batch_size, self.num_heads, tgt_time_steps, key.size(0))  # qk.size = [batch_size, num_heads, tgt_time_steps, src_time_steps]
            qk = qk.masked_fill(key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2), float('-inf'))  # qk.size = [batch_size, num_heads, tgt_time_steps, src_time_steps]
            qk = qk.view(batch_size * self.num_heads, tgt_time_steps, key.size(0))  # qk.size = [batch_size * num_heads, tgt_time_steps, src_time_steps]
        if attn_mask is not None:
            qk += attn_mask    # qk.size = [batch_size * num_heads, tgt_time_steps, src_time_steps]
        
        qk_scale = qk / self.head_scaling    # qk_scale.size = [batch_size * num_heads, tgt_time_steps, src_time_steps]
        qk_scale_softmax = F.softmax(qk_scale, dim=-1)    # qk_scale_softmax.size = [batch_size * num_heads, tgt_time_steps, src_time_steps]

        if need_weights:
            # attn_weights.size = [num_heads, batch_size, tgt_time_steps, src_time_steps]
            attn_weights = qk_scale_softmax.view(self.num_heads, batch_size, tgt_time_steps, key.size(0))  
        else: attn_weights = None

        out = torch.bmm(qk_scale_softmax, parallel_values)    # out.size = [batch_size * num_heads, tgt_time_steps, head_embed_size]
        out = out.view(batch_size, self.num_heads, tgt_time_steps, self.head_embed_size)  # out.size = [batch_size, num_heads, tgt_time_steps, head_embed_size]
        out = out.transpose(1, 2)    # out.size = [batch_size, tgt_time_steps, num_heads, head_embed_size]
        out = out.contiguous().view(batch_size, tgt_time_steps, self.embed_dim)  # out.size = [batch_size, tgt_time_steps, embed_dim]
        out = out.transpose(0, 1)    # out.size = [tgt_time_steps, batch_size, embed_dim]
        attn = self.out_proj(out)    # attn.size = [tgt_time_steps, batch_size, embed_dim]
        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
