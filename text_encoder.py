import os
import torch
from torch import nn
from torch.nn import functional as func
from torch.nn.utils import rnn
from torchvision import models
from gensim.models.keyedvectors import KeyedVectors



class IngrEmbedLayer(nn.Module):
    def __init__(self, data_dir, emb_dim):
        super(IngrEmbedLayer, self).__init__()
        path = os.path.join(data_dir, 'vocab_ingr.txt')
        with open(path, 'r') as f:
            num_ingr = len(f.read().split('\n'))
            print('num_ingr = ', num_ingr)
        # first three index has special meaning, see utils.py
        emb = nn.Embedding(35549, emb_dim, padding_idx=0)
        # emb = nn.Embedding(num_ingr+3, emb_dim, padding_idx=0)
        self.embed_layer = emb
        print('==> Ingr embed layer', emb)

    def forward(self, sent_list):  # 1992, 300
        # sent_list [BS, max_len]  16, 20
        return self.embed_layer(sent_list)  # x=[BS, max_len, emb_dim]

class SentEncoder(nn.Module):
    def __init__(
            self,
            data_dir,
            emb_dim,
            hid_dim,
            with_attention=True):
        super(SentEncoder, self).__init__()
        self.embed_layer = IngrEmbedLayer(data_dir=data_dir, emb_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,  # 300
            hidden_size=hid_dim,  # 300
            bidirectional=True,
            batch_first=True)
        if with_attention:
            self.atten_layer = AttentionLayer( 2 *hid_dim)
        self.with_attention = with_attention

    def forward(self, sent_list):
        # sent_list [BS, max_len]
        x = self.embed_layer(sent_list)  # x=[BS, max_len, emb_dim]
        # print(sent_list.shape)
        # lens = (sent_list==1).nonzero()[:,1] + 1
        lens = sent_list.count_nonzero(dim=1) + 1
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        # print(lens.shape) = batch size = 32
        sorted_len, sorted_idx = lens.sort(0, descending=True)  # sorted_idx=[BS], for sorting
        # print('sorted len: ', sorted_len)
        _, original_idx = sorted_idx.sort(0, descending=False)  # original_idx=[BS], for unsorting
        # print(sorted_idx.shape, x.shape)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)  # sorted_idx=[BS, max_len, emb_dim]
        sorted_inputs = x.gather(0, index_sorted_idx.long())  # sort by num_words
        # print(sorted_inputs.shape)
        packed_seq = rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        # 長さの異なる系列をミニバッチ化する際には，長さを揃えるためにpaddingが必要
        # print(packed_seq.batch_sizes)

        if self.with_attention:
            out, _ = self.rnn(packed_seq)
            # RuntimeError: start (328) + length (2) exceeds dimension size (328).
            # except RuntimeError:
            #     print(packed_seq)
            #    print()
            #    print(packed_seq.batch_sizes)
            #    tensor([32, 32, 32, 32, 31, 28, 27, 21, 20, 17, 15, 13,  5,  5,  4,  4,  3,  3, 2,  2,  2]) = 330>328
            y, _ = rnn.pad_packed_sequence(out, batch_first=True)
            # y=[BS, max_len, 2*hid_dim], currently in WRONG order!
            unsorted_idx = original_idx.view(-1 ,1 ,1).expand_as(y)
            output = y.gather(0, unsorted_idx).contiguous() # [BS, max_len, 2*hid_dim], now in correct order
            feat = self.atten_layer(output)
        else:
            _, (h ,_) = self.rnn(packed_seq) # [2, BS, hid_dim], currently in WRONG order!
            h = h.transpose(0 ,1) # [BS, 2, hid_dim], still in WRONG order!
            # unsort the output
            unsorted_idx = original_idx.view(-1 ,1 ,1).expand_as(h)
            output = h.gather(0, unsorted_idx).contiguous()  # [BS, 2, hid_dim], now in correct order
            feat = output.view(output.size(0), output.size(1 ) *output.size(2))  # [BS, 2*hid_dim]

        # print('sent', feat.shape) # [BS, 2*hid_dim]
        return feat


class SentEncoderFC(nn.Module):
    def __init__(
            self,
            data_dir,
            emb_dim,
            hid_dim,
            with_attention=True):
        super(SentEncoderFC, self).__init__()
        self.embed_layer = IngrEmbedLayer(data_dir=data_dir, emb_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, 2* hid_dim)
        if with_attention:
            self.atten_layer = AttentionLayer(2 * hid_dim)
        self.with_attention = with_attention

    def forward(self, sent_list):
        # sent_list [BS, max_len]
        x = self.embed_layer(sent_list)  # x=[BS, max_len, emb_dim]
        x = self.fc(x)  # [BS, max_len, 2*hid_dim]
        if not self.with_attention:
            feat = x.sum(dim=1)  # [BS, 2*hid_dim]
        else:
            feat = self.atten_layer(x)  # [BS, 2*hid_dim]
        # print('ingredients', feat.shape)
        return feat


class TextEncoder(nn.Module):
    def __init__(
            self, data_dir,  hid_dim, emb_dim, z_dim, with_attention, ingr_enc_type='fc'):
        super(TextEncoder, self).__init__()
        if ingr_enc_type == 'rnn':
            self.ingr_encoder = SentEncoder(
                data_dir,
                emb_dim,
                hid_dim,
                with_attention)
            self.ingr_encoder.to('cuda')
            # emb_dim=ckpt_args.word2vec_dim,  # 300
            #         hid_dim=ckpt_args.rnn_hid_dim,  # 300
            #         z_dim=ckpt_args.feature_dim,  # 1024
        elif ingr_enc_type == 'fc':
            self.ingr_encoder = SentEncoderFC(
                data_dir,
                emb_dim,
                hid_dim,
                with_attention)
            self.ingr_encoder.to('cuda')
        self.bn = nn.BatchNorm1d(4 * hid_dim)  # hid_dim = 300
        self.fc = nn.Linear(4 * hid_dim, z_dim)  # z_dim = 128 //1024
        self.bn.cuda()
        self.fc.cuda()

    def forward(self, recipe_list):

        title_list = recipe_list[0]
        ingredients_list = recipe_list[1]

        # instructions_list = recipe_list[0][4]
        '''     
        t0 = torch.zeros(32, 4).cuda()
        title_list = torch.concat((recipe_list[0][0], t0), 1)
        ingredients_list = torch.concat((recipe_list[0][2], t0), 1)
        instructions_tmp = recipe_list[0][4]
        t1 = torch.zeros(20, 4).cuda()
        instructions_list = torch.zeros(32, 24, 24)
        for i in range(0, 32):
            instructions_list[i] = torch.concat((instructions_tmp[i], t1), 1)

            #x = torch.concat((recipe_list[0][4][i], t1), 1)
            #instructions_list = torch.stack([ingredients_list, x], 0)
            t2 = torch.zeros(4, 24).cuda()
            instructions_list = torch.concat((ingredients_list, t2), 0)

        instructions_list = torch.concat((recipe_list[0][4][0], t1), 1)
        '''

        # instructions_list = torch.concat(recipe_list[0][4], 1)


        # title_list, ingredients_list, instructions_list = recipe_list
        # if self.text_info == '111':
        #     feat_title = self.sent_encoder(title_list)
        #     feat_ingredients = self.ingr_encoder(ingredients_list)
        #     feat_instructions = self.doc_encoder(instructions_list)
        #     feat = torch.cat([feat_title, feat_ingredients, feat_instructions], dim=1)
        #     feat = torch.tanh(self.fc(self.bn(feat)))
        # elif self.text_info == '010':
        #     feat_ingredients = self.ingr_encoder(ingredients_list)
        #     feat = torch.tanh(self.fc(self.bn(feat_ingredients)))
        feat_title = self.ingr_encoder(title_list)
        feat_ingredients = self.ingr_encoder(ingredients_list)
        feat = torch.cat([feat_title, feat_ingredients], dim=1)
        feat = torch.tanh(self.fc(self.bn(feat)))
        # print('recipe', feat.shape)
        return feat


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.u = torch.nn.Parameter(torch.randn(input_dim))  # u = [2*hid_dim] a shared contextual vector
        # torch.randn 平均が 0 で分散が 1 の正規分布（標準正規分布とも呼ばれます）から乱数で満たされたテンソルを返す
        # torch.nn.parameter レイヤーのパラメータとして定義する
        self.u.requires_grad = True
        self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        # x = [BS, num_vec, 2*hid_dim]
        mask = (x!=0)
        # a trick used to find the mask for the softmax
        mask = mask[:,:,0].bool()
        h = torch.tanh(self.fc(x))  # h = [BS, num_vec, 2*hid_dim]
        tmp = h @ self.u  # tmp = [BS, num_vec], unnormalized importance
        masked_tmp = tmp.masked_fill(~mask, -1e32)
        alpha = func.softmax(masked_tmp, dim=1)  # alpha = [BS, num_vec], normalized importance
        alpha = alpha.unsqueeze(-1)  # alpha = [BS, num_vec, 1]
        out = x * alpha  # out = [BS, num_vec, 2*hid_dim]
        out = out.sum(dim=1)  # out = [BS, 2*hid_dim]
        # pdb.set_trace()
        return out


def create_model(ckpt_args, device='cuda'):
    param_counter = lambda params: sum(p.numel() for p in params if p.requires_grad)
    text_encoder = TextEncoder(
        # self, data_dir, text_info, hid_dim, emb_dim, z_dim, with_attention, ingr_enc_type
        data_dir= 'E:/CookGAN/retrieval_model/models',
        emb_dim=ckpt_args.word2vec_dim,  # 300
        hid_dim=ckpt_args.rnn_hid_dim,  # 300
        z_dim=ckpt_args.feature_dim,  # 1024
        # word2vec_file=ckpt_args.word2vec_file,
        text_info=ckpt_args.text_info,
        with_attention=ckpt_args.with_attention,
        ingr_enc_type=ckpt_args.ingrs_enc_type)
    # image_encoder = ImageEncoder(
    #     z_dim=ckpt_args.feature_dim)
    text_encoder.to(device)
    print('# text_encoder', param_counter(text_encoder.parameters()))
    # print('# image_encoder', param_counter(image_encoder.parameters()))
    if device == 'cuda':
        # text_encoder, image_encoder = [nn.DataParallel(x) for x in [text_encoder, image_encoder]]
        text_encoder = nn.DataParallel(text_encoder)
    optimizer = torch.optim.Adam([
            {'params': text_encoder.parameters()},
            #{'params': image_encoder.parameters()},
        ], lr=ckpt_args.lr, betas=(0.5, 0.999))
    return text_encoder, optimizer

def load_model(ckpt_path, device='cuda'):  # word2vec_recipes.bi
    print('load retrieval model from:', ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    batch_idx = ckpt['batch_idx']
    text_encoder, optimizer = create_model(ckpt_args, device)
    if device=='cpu':
        text_encoder.load_state_dict(ckpt['text_encoder'])
        # image_encoder.load_state_dict(ckpt['image_encoder'])
    else:
        text_encoder.module.load_state_dict(ckpt['text_encoder'])
        # image_encoder.module.load_state_dict(ckpt['image_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt_args, batch_idx, text_encoder, optimizer

