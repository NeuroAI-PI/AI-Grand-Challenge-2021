import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import ElectraModel
from torch.autograd import Variable
import math


class ElectraForSequenceClassification2(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(ElectraForSequenceClassification2, self).__init__()
        self.select_layers_len = select_layers_len
        self.dense = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len)
        self.out_proj = nn.Linear(config.hidden_size * self.select_layers_len, num_labels)

        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        model_input = self.dropout(model_input)
        output = self.leakyrelu(self.dense(model_input))
        output = self.dropout(output)
        logits = self.out_proj(output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class ElectraForSequenceClassification3(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(ElectraForSequenceClassification3, self).__init__()
        self.select_layers_len = select_layers_len
        self.dense = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len)
        self.out_proj1 = nn.Linear(config.hidden_size * self.select_layers_len, config.hidden_size * self.select_layers_len // 2)
        self.dense2 = nn.Linear(config.hidden_size * self.select_layers_len // 2, config.hidden_size * self.select_layers_len // 2)
        self.out_proj2 = nn.Linear(config.hidden_size * self.select_layers_len // 2, num_labels)

        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        model_input = self.dropout(model_input)
        output = self.gelu(self.dense(model_input))

        output = self.dropout(output)
        output = self.out_proj1(output)

        output = self.dropout(output)
        output = self.gelu(self.dense2(output))

        output = self.dropout(output)
        logits = self.out_proj2(output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            # logits = self.softmax(logits)  # 우리 test 결과도 softmax 안썼으니
            return logits


class ElectraForSequenceClassificationMLP1(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(ElectraForSequenceClassificationMLP1, self).__init__()
        self.select_layers_len = select_layers_len
        self.classifier = nn.Linear(config.hidden_size*self.select_layers_len, num_labels)

        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        model_input = self.dropout(model_input)
        logits = self.classifier(model_input)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            # logits = self.softmax(logits)  # 우리 test 결과도 softmax 안썼으니
            return logits


class ElectraForSequenceClassificationMLP2(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(ElectraForSequenceClassificationMLP2, self).__init__()
        self.select_layers_len = select_layers_len
        self.fc1 = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len // 2)
        self.fc2 = nn.Linear(config.hidden_size*self.select_layers_len // 2, num_labels)

        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        out = self.dropout(model_input)
        out = self.leakyrelu(self.fc1(out))
        logits = self.fc2(out)

        # model_input = self.leakyrelu(model_input)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            # logits = self.softmax(logits)  # 우리 test 결과도 softmax 안썼으니
            return logits


class ElectraForSequenceClassificationMLP3(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(ElectraForSequenceClassificationMLP3, self).__init__()
        self.select_layers_len = select_layers_len
        self.fc1 = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len // 2)
        self.fc2 = nn.Linear(config.hidden_size * self.select_layers_len // 2, config.hidden_size * self.select_layers_len // 4)
        self.fc3 = nn.Linear(config.hidden_size * self.select_layers_len // 4, num_labels)

        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        out = self.dropout(model_input)
        out = self.fc1(out)
        out = self.fc2(out)
        logits = self.fc3(out)

        # model_input = self.leakyrelu(model_input)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            # logits = self.softmax(logits)  # 우리 test 결과도 softmax 안썼으니
            return logits


class BertForSequenceClassificationEnsembleMLP(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(BertForSequenceClassificationEnsembleMLP, self).__init__()
        self.select_layers_len = select_layers_len
        self.electra = ElectraModel.from_pretrained(model_name, config=config)

        # ensemble model 1
        self.e1_dense = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len)
        self.e1_out_proj = nn.Linear(config.hidden_size * self.select_layers_len, num_labels)

        # ensemble model 2
        self.e2_dense = nn.Linear(config.hidden_size*self.select_layers_len, config.hidden_size*self.select_layers_len)
        self.e2_out_proj = nn.Linear(config.hidden_size * self.select_layers_len, num_labels)

        self.gelu = nn.GELU()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1]) - self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i][:, 0, :]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        # ensemble model 1
        model_input = output_cls_reshape
        model_input = self.dropout(model_input)
        e1_out = self.gelu(self.e1_dense(model_input))
        e1_out = self.dropout(e1_out)
        e1_logits = self.e1_out_proj(e1_out)

        del e1_out
        # ensemble model 2
        e2_out = self.leakyrelu(self.e2_dense(model_input))
        e2_out = self.dropout(e2_out)
        e2_logits = self.e2_out_proj(e2_out)

        del e2_out

        logits = torch.stack([e1_logits, e2_logits], dim=1)
        logits = torch.sum(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassificationBiLSTM(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(BertForSequenceClassificationBiLSTM, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name, config=config)

        self.num_labels = num_labels
        self.input_size = 256
        self.rnn_hidden = 768
        self.select_layers_len = 4
        self.num_layers = 2

        self.leakyrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)

        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.lstm_1 = nn.LSTM(config.hidden_size, config.hidden_size // 8, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        # self.relu_1 = nn.ReLU()

        self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
        self.lstm_2 = nn.LSTM(config.hidden_size, config.hidden_size // 8, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        # self.relu_2 = nn.ReLU()

        self.dropout_3 = nn.Dropout(config.hidden_dropout_prob)
        self.lstm_3 = nn.LSTM(config.hidden_size, config.hidden_size // 8, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        # self.relu_3 = nn.ReLU()

        self.dropout_4 = nn.Dropout(config.hidden_dropout_prob)
        self.lstm_4 = nn.LSTM(config.hidden_size, config.hidden_size // 8, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True, dropout=config.hidden_dropout_prob)
        # self.relu_4 = nn.ReLU()

        # self.fc1 = nn.Linear(self.input_size * 768, self.input_size * 768 // 16)
        # self.fc2 = nn.Linear(self.input_size * 768 // 16, self.num_labels)

        self.fc1 = nn.Linear(self.input_size * 768, self.input_size * 768 // 512)
        self.fc2 = nn.Linear(self.input_size * 768 // 512, self.num_labels)

        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.01)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        # # version 1
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        outputs = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]

        out = outputs[-1]
        out = self.dropout_1(out)
        out, _ = self.lstm_1(out)

        out2 = outputs[-2]
        out2 = self.dropout_2(out2)
        out2, _ = self.lstm_2(out2)

        out = torch.cat((out, out2), 2)

        out2 = outputs[-3]
        out2 = self.dropout_3(out2)
        out2, _ = self.lstm_3(out2)
        out = torch.cat((out, out2), 2)

        out2 = outputs[-4]

        out2 = self.dropout_4(out2)
        out2, _ = self.lstm_4(out2)
        out = torch.cat((out, out2), 2)

        del out2, outputs

        out = out.reshape([out.shape[0], -1])

        out = self.fc1(out)
        logits = self.fc2(out)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassificationLayerNormGRU(nn.Module):
    def __init__(self, model_name, config, select_layers_len, num_labels=5):
        super(BertForSequenceClassificationLayerNormGRU, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name, config=config)

        self.hidden_dim = config.hidden_size
        self.input_size = 512
        # Number of hidden layers
        self.layer_dim = 1
        self.select_layers_len = 1

        self.gru_cell = LayerNormGRUCell(self.hidden_dim, self.hidden_dim, self.layer_dim)
        self.fc = nn.Linear(self.hidden_dim, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        output_hidden_states = self.electra(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        output_select_encoded_layer = output_hidden_states[1][len(output_hidden_states[1])-self.select_layers_len:]
        for i in range(self.select_layers_len):
            output_cls_layers = output_select_encoded_layer[i]
            if i == 0:
                output_cls_reshape = output_cls_layers
            else:
                output_cls_reshape = torch.cat((output_cls_layers, output_cls_layers), 1)

        model_input = output_cls_reshape
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28

        h0 = Variable(torch.zeros(self.layer_dim, model_input.size(0), self.hidden_dim).cuda())
        hn, outs = h0[0, :, :], []

        for seq in range(model_input.size(1)):
            hn = self.gru_cell(model_input[:, seq, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)

        return out


class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t


