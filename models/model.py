import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from torch.nn.utils import clip_grad_norm

from .data_utils import data_utils
from .modules import mlp


class InverseSquareRootSchedule(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_updates, last_epoch=-1):
        lr_step = 1.0 / warmup_updates
        decay_factor = warmup_updates ** 0.5

        def lr_lambda(num_updates):
            if num_updates < warmup_updates:
                return num_updates * lr_step
            else:
                return decay_factor * (num_updates ** -0.5)

        super(InverseSquareRootSchedule, self).__init__(optimizer, lr_lambda, last_epoch)


class PlotCodeGenerator(nn.Module):
    def __init__(self, args, word_vocab_size, code_vocab):
        super(PlotCodeGenerator, self).__init__()
        self.cuda_flag = args.cuda
        self.word_vocab_size = word_vocab_size
        self.code_vocab_size = args.code_vocab_size
        self.num_plot_types = args.num_plot_types
        self.code_vocab = code_vocab
        self.batch_size = args.batch_size
        self.embedding_size = args.embedding_size
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.MLP_hidden_size = args.MLP_hidden_size
        self.num_LSTM_layers = args.num_LSTM_layers
        self.num_MLP_layers = args.num_MLP_layers
        self.gradient_clip = args.gradient_clip
        self.lr = args.lr
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_eps = args.adam_eps
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        self.dropout_rate = args.dropout_rate
        self.nl = args.nl
        self.use_comments = args.use_comments
        self.code_context = args.code_context
        self.bow = args.bow
        self.hierarchy = args.hierarchy
        self.label_prediction = args.label_prediction
        self.copy_mechanism = args.copy_mechanism
        self.nl_code_linking = args.nl_code_linking
        self.max_word_len = args.max_word_len
        self.max_code_context_len = args.max_code_context_len
        self.max_decode_len = args.max_decode_len
        self.freeze_nl_embeddings = args.freeze_nl_embeddings
        self.freeze_nl_encoder_first_n = args.freeze_nl_encoder_first_n
        self.freeze_nl_pooler = args.freeze_nl_pooler
        self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.bow:
            self.input_size = self.word_vocab_size + self.code_vocab_size
            self.classifier = mlp.MLPModel(self.num_MLP_layers, self.input_size, self.MLP_hidden_size,
                                           self.num_plot_types, self.dropout_rate, self.cuda_flag)
        else:
            if self.copy_mechanism:
                self.code_embedding = nn.Embedding(self.code_vocab_size, self.embedding_size)
            else:
                self.code_embedding = nn.Embedding(self.code_vocab_size + self.max_code_context_len,
                                                   self.embedding_size)
                self.code_predictor = nn.Linear(self.embedding_size, self.code_vocab_size + self.max_code_context_len)
                self.copy_predictor = nn.Linear(self.embedding_size, self.code_vocab_size + self.max_code_context_len)

            self.input_nl_encoder = transformers.AutoModel.from_pretrained(
                args.nl_model_name,
                cache_dir=args.transformers_cache_dir
            )
            self.input_nl_encoder.resize_token_embeddings(self.word_vocab_size)

            self.input_code_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size,
                                              num_layers=self.num_LSTM_layers, dropout=self.dropout_rate,
                                              batch_first=True, bidirectional=True)
            if self.hierarchy:
                self.decoder = nn.LSTM(input_size=self.embedding_size * 2, hidden_size=self.LSTM_hidden_size,
                                       num_layers=self.num_LSTM_layers, dropout=self.dropout_rate,
                                       batch_first=True, bidirectional=True)
            else:
                self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size,
                                       num_layers=self.num_LSTM_layers, dropout=self.dropout_rate,
                                       batch_first=True, bidirectional=True)
            self.word_attention = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
            if not self.nl_code_linking:
                self.code_ctx_linear = nn.Linear(self.LSTM_hidden_size * 2 + self.embedding_size, self.embedding_size)
            else:
                self.code_ctx_word_linear = nn.Linear(self.LSTM_hidden_size * 4 + self.embedding_size,
                                                      self.embedding_size)
                self.code_word_linear = nn.Linear(self.LSTM_hidden_size * 2 + self.embedding_size, self.embedding_size)
            if self.label_prediction:
                self.plot_type_classifier = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4,
                                                         self.MLP_hidden_size, self.num_plot_types, self.dropout_rate,
                                                         self.cuda_flag)
                self.plot_type_embedding = nn.Embedding(self.num_plot_types, self.embedding_size)
            self.encoder_code_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)
            self.decoder_code_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)
            self.decoder_copy_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)
            self.encoder_copy_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)
            self.target_embedding_linear = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)

        # training
        self.loss = nn.CrossEntropyLoss()

        parameters_to_optimize = [param for param in self.parameters() if param.requires_grad]
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                parameters_to_optimize,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
                weight_decay=self.weight_decay
            )
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                parameters_to_optimize,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                parameters_to_optimize,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                parameters_to_optimize,
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)
        self.scheduler = InverseSquareRootSchedule(self.optimizer, warmup_updates=self.warmup_steps)

    def init_weights(self, param_init):
        for name, param in self.named_parameters():
            name_split = name.split(".")
            if name_split[0] != "input_nl_encoder":
                nn.init.uniform_(param, -param_init, param_init)
            else:
                if name_split[1] == "embeddings":
                    if self.freeze_nl_embeddings:
                        param.requires_grad_(False)
                elif name_split[1] == "encoder":
                    assert name_split[2] == "layer"
                    layer_id = int(name_split[3])
                    if layer_id < self.freeze_nl_encoder_first_n:
                        param.requires_grad_(False)
                elif name_split[1] == "pooler":
                    if self.freeze_nl_pooler:
                        param.requires_grad_(False)
                else:
                    raise ValueError()

    def train_step(self):
        if self.gradient_clip is not None:
            clip_grad_norm([param for param in self.parameters() if param.requires_grad], self.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

    def forward(self, batch_input, batch_labels, eval_flag=False):
        batch_size = batch_labels.size()[0]
        if self.bow:
            pred_logits = self.classifier(batch_input)
            avg_loss = self.loss(pred_logits, batch_labels)
            return avg_loss, pred_logits
        else:
            batch_init_data = batch_input['init_data']
            batch_nl_input = batch_input['nl']
            # encoder_word_mask = (batch_nl_input == data_utils.PAD_ID).float()
            # encoder_word_mask = torch.max(encoder_word_mask, (batch_nl_input == data_utils.UNK_ID).float())
            # encoder_word_mask = torch.max(encoder_word_mask, (batch_nl_input == data_utils.EOS_ID).float())
            # if self.cuda_flag:
            # 	encoder_word_mask = encoder_word_mask.cuda()
            nl_encoder_output, _ = self.input_nl_encoder(**batch_nl_input)
            encoder_word_mask = torch.where(
                batch_nl_input["attention_mask"].byte(),
                torch.tensor(0.0, dtype=nl_encoder_output.dtype, device=nl_encoder_output.device),
                torch.tensor(-1e9, dtype=nl_encoder_output.dtype, device=nl_encoder_output.device)
            )
            # decoder_hidden_state = nl_hidden_state

            batch_code_context_input = batch_input['code_context']
            batch_code_context_embedding = self.code_embedding(batch_code_context_input)
            batch_code_nl_embedding = []
            batch_input_code_nl_indices = batch_input['input_code_nl_indices']
            # max_code_len = batch_code_context_input.size()[1]
            # max_word_len = batch_nl_input.size()[1]

            if self.nl_code_linking:
                for batch_idx in range(batch_size):
                    input_code_nl_indices = batch_input_code_nl_indices[batch_idx, :, :]
                    cur_code_nl_embedding_0 = nl_encoder_output[batch_idx, input_code_nl_indices[:, 0]]
                    cur_code_nl_embedding_1 = nl_encoder_output[batch_idx, input_code_nl_indices[:, 1]]
                    cur_code_nl_embedding = cur_code_nl_embedding_0 + cur_code_nl_embedding_1
                    batch_code_nl_embedding.append(cur_code_nl_embedding)
                batch_code_nl_embedding = torch.stack(batch_code_nl_embedding, dim=0)
                code_encoder_input = torch.cat([batch_code_context_embedding, batch_code_nl_embedding], dim=-1)
                code_encoder_input = self.code_word_linear(code_encoder_input)
            else:
                code_encoder_input = batch_code_context_embedding

            # encoder_code_mask = (batch_code_context_input == data_utils.PAD_ID).float()
            # encoder_code_mask = torch.max(encoder_code_mask, (batch_code_context_input == data_utils.UNK_ID).float())
            # encoder_code_mask = torch.max(encoder_code_mask, (batch_code_context_input == data_utils.EOS_ID).float())
            # if self.cuda_flag:
            # 	encoder_code_mask = encoder_code_mask.cuda()
            code_encoder_output, code_hidden_state = self.input_code_encoder(code_encoder_input)
            decoder_hidden_state = code_hidden_state

            gt_output = batch_input['gt']
            target_code_output = batch_input['code_output']
            target_df_output = batch_input['df_output']
            target_var_output = batch_input['var_output']
            target_str_output = batch_input['str_output']
            code_output_mask = batch_input['code_output_mask']
            output_df_mask = batch_input['output_df_mask']
            output_var_mask = batch_input['output_var_mask']
            output_str_mask = batch_input['output_str_mask']
            if not eval_flag:
                decode_length = target_code_output.size()[1]
            else:
                '''
                decode_length = self.max_decode_len
                '''
                decode_length = target_code_output.size()[1]

            if self.label_prediction:
                plot_type_classifier_input = torch.cat([nl_encoder_output[:, -1, :], code_encoder_output[:, -1, :]],
                                                       dim=-1)
                plot_type_classification_logits = self.plot_type_classifier(plot_type_classifier_input)
                if eval_flag:
                    decoder_input = plot_type_classification_logits.max(1)[1]
                else:
                    decoder_input = batch_labels
                decoder_input_embedding = self.plot_type_embedding(decoder_input)
                decoder_input_embedding = decoder_input_embedding.unsqueeze(1)
            else:
                decoder_input_sketch = torch.ones(batch_size, 1, dtype=torch.int64) * data_utils.GO_ID
                if self.cuda_flag:
                    decoder_input_sketch = decoder_input_sketch.cuda()
                decoder_input_sketch_embedding = self.code_embedding(decoder_input_sketch)
                decoder_input = torch.ones(batch_size, 1, dtype=torch.int64) * data_utils.GO_ID
                if self.cuda_flag:
                    decoder_input = decoder_input.cuda()
                decoder_input_embedding = self.code_embedding(decoder_input)
            finished = torch.zeros(batch_size, 1, dtype=torch.int64)

            max_code_mask_len = code_output_mask.size()[1]

            pad_mask = torch.zeros(max_code_mask_len)
            pad_mask[data_utils.PAD_ID] = 1e9
            pad_mask = torch.stack([pad_mask] * batch_size, dim=0)
            if self.cuda_flag:
                finished = finished.cuda()
                pad_mask = pad_mask.cuda()

            batch_code_output_indices = data_utils.np_to_tensor(np.array(list(range(self.code_vocab_size))), 'int',
                                                                self.cuda_flag)
            batch_code_output_embedding = self.code_embedding(batch_code_output_indices)
            batch_code_output_embedding = torch.stack([batch_code_output_embedding] * batch_size, dim=0)

            batch_output_code_ctx_embedding = []
            batch_output_code_ctx_indices = batch_input['output_code_ctx_indices']
            for batch_idx in range(batch_size):
                output_code_ctx_indices = batch_output_code_ctx_indices[batch_idx]
                cur_output_code_ctx_embedding = code_encoder_output[batch_idx, output_code_ctx_indices]
                batch_output_code_ctx_embedding.append(cur_output_code_ctx_embedding)
            batch_output_code_ctx_embedding = torch.stack(batch_output_code_ctx_embedding, dim=0)

            if self.nl_code_linking:
                batch_output_code_nl_embedding = []
                batch_output_code_nl_indices = batch_input['output_code_nl_indices']
                for batch_idx in range(batch_size):
                    output_code_nl_indices = batch_output_code_nl_indices[batch_idx, :, :]
                    cur_output_code_nl_embedding_0 = nl_encoder_output[batch_idx, output_code_nl_indices[:, 0]]
                    cur_output_code_nl_embedding_1 = nl_encoder_output[batch_idx, output_code_nl_indices[:, 1]]
                    cur_output_code_nl_embedding = cur_output_code_nl_embedding_0 + cur_output_code_nl_embedding_1
                    batch_output_code_nl_embedding.append(cur_output_code_nl_embedding)
                batch_output_code_nl_embedding = torch.stack(batch_output_code_nl_embedding, dim=0)
                batch_code_output_embedding = torch.cat(
                    [batch_code_output_embedding, batch_output_code_ctx_embedding, batch_output_code_nl_embedding],
                    dim=-1)
                batch_code_output_embedding = self.code_ctx_word_linear(batch_code_output_embedding)
            else:
                batch_code_output_embedding = torch.cat([batch_code_output_embedding, batch_output_code_ctx_embedding],
                                                        dim=-1)
                batch_code_output_embedding = self.code_ctx_linear(batch_code_output_embedding)

            if self.code_context:
                batch_code_output_context_embedding = []

                for batch_idx in range(batch_size):
                    output_code_indices = batch_init_data[batch_idx]['output_code_indices']
                    cur_code_output_context_embedding = []
                    for code_idx in output_code_indices:
                        cur_code_output_context_embedding.append(code_encoder_output[batch_idx, code_idx, :])
                    if len(cur_code_output_context_embedding) < max_code_mask_len - self.code_vocab_size:
                        cur_code_output_context_embedding += [data_utils.np_to_tensor(
                            np.zeros(self.LSTM_hidden_size * 2), 'float', self.cuda_flag)] * (
                                                                 max_code_mask_len - self.code_vocab_size - len(
                                                                 cur_code_output_context_embedding))
                    cur_code_output_context_embedding = torch.stack(cur_code_output_context_embedding, dim=0)
                    batch_code_output_context_embedding.append(cur_code_output_context_embedding)
                batch_code_output_context_embedding = torch.stack(batch_code_output_context_embedding, dim=0)
                batch_code_output_context_embedding = self.target_embedding_linear(batch_code_output_context_embedding)
                batch_code_output_embedding = torch.cat(
                    [batch_code_output_embedding, batch_code_output_context_embedding], dim=1)

            code_pred_logits = []
            code_predictions = []
            df_pred_logits = []
            df_predictions = []
            var_pred_logits = []
            var_predictions = []
            str_pred_logits = []
            str_predictions = []
            predictions = []

            for step in range(decode_length):
                if self.hierarchy:
                    decoder_output, decoder_hidden_state = self.decoder(
                        torch.cat([decoder_input_sketch_embedding, decoder_input_embedding], dim=-1),
                        decoder_hidden_state)
                else:
                    decoder_output, decoder_hidden_state = self.decoder(decoder_input_embedding, decoder_hidden_state)
                decoder_output = decoder_output.squeeze(1)

                decoder_nl_attention = self.word_attention(decoder_output)
                attention_logits = torch.bmm(nl_encoder_output, decoder_nl_attention.unsqueeze(2))
                attention_logits = attention_logits.squeeze(-1)
                attention_logits = attention_logits + encoder_word_mask
                attention_weights = nn.Softmax(dim=-1)(attention_logits)
                attention_weights = self.dropout(attention_weights)
                nl_attention_vector = torch.bmm(torch.transpose(nl_encoder_output, 1, 2),
                                                attention_weights.unsqueeze(2))
                nl_attention_vector = nl_attention_vector.squeeze(-1)

                input_code_encoding = self.encoder_code_attention_linear(nl_attention_vector)
                if self.hierarchy:
                    input_copy_encoding = self.encoder_copy_attention_linear(nl_attention_vector)

                decoder_code_output = self.decoder_code_attention_linear(decoder_output)
                if self.hierarchy:
                    decoder_copy_output = self.decoder_copy_attention_linear(decoder_output)

                decoder_code_output = decoder_code_output + input_code_encoding
                if self.hierarchy:
                    decoder_copy_output = decoder_copy_output + input_copy_encoding

                if self.copy_mechanism:
                    cur_code_pred_logits = torch.bmm(batch_code_output_embedding, decoder_code_output.unsqueeze(2))
                    cur_code_pred_logits = cur_code_pred_logits.squeeze(-1)
                else:
                    cur_code_pred_logits = self.code_predictor(decoder_code_output)
                cur_code_pred_logits = cur_code_pred_logits + finished.float() * pad_mask
                cur_code_pred_logits = cur_code_pred_logits - (1.0 - code_output_mask) * 1e9
                cur_code_predictions = cur_code_pred_logits.max(1)[1]

                if eval_flag:
                    cur_code_predictions = cur_code_pred_logits.max(1)[1]
                    sketch_predictions = cur_code_predictions
                else:
                    cur_code_predictions = target_code_output[:, step]
                    sketch_predictions = target_code_output[:, step]

                if self.hierarchy:
                    if self.copy_mechanism:
                        cur_copy_pred_logits = torch.bmm(batch_code_output_embedding, decoder_copy_output.unsqueeze(2))
                        cur_copy_pred_logits = cur_copy_pred_logits.squeeze(-1)
                    else:
                        cur_copy_pred_logits = self.copy_predictor(decoder_copy_output)
                    cur_df_pred_logits = cur_copy_pred_logits - (1.0 - output_df_mask) * 1e9
                    cur_df_predictions = cur_df_pred_logits.max(1)[1] * (
                        (sketch_predictions == data_utils.DF_ID).long())

                    cur_var_pred_logits = cur_copy_pred_logits - (1.0 - output_var_mask) * 1e9
                    cur_var_predictions = cur_var_pred_logits.max(1)[1] * (
                        (sketch_predictions == data_utils.VAR_ID).long())

                    cur_str_pred_logits = cur_copy_pred_logits - (1.0 - output_str_mask) * 1e9
                    cur_str_predictions = cur_str_pred_logits.max(1)[1] * (
                        (sketch_predictions == data_utils.STR_ID).long())

                if eval_flag:
                    decoder_input_sketch = cur_code_predictions
                    decoder_input = cur_code_predictions
                    if self.hierarchy:
                        decoder_input = torch.max(decoder_input, cur_df_predictions)
                        decoder_input = torch.max(decoder_input, cur_var_predictions)
                        decoder_input = torch.max(decoder_input, cur_str_predictions)
                else:
                    decoder_input_sketch = target_code_output[:, step]
                    decoder_input = gt_output[:, step]
                if self.copy_mechanism:
                    decoder_input_sketch_embedding = []
                    for batch_idx in range(batch_size):
                        decoder_input_sketch_embedding.append(
                            batch_code_output_embedding[batch_idx, decoder_input_sketch[batch_idx], :])
                    decoder_input_sketch_embedding = torch.stack(decoder_input_sketch_embedding, dim=0)

                    decoder_input_embedding = []
                    for batch_idx in range(batch_size):
                        decoder_input_embedding.append(
                            batch_code_output_embedding[batch_idx, decoder_input[batch_idx], :])
                    decoder_input_embedding = torch.stack(decoder_input_embedding, dim=0)
                else:
                    decoder_input_sketch_embedding = self.code_embedding(decoder_input_sketch)
                    decoder_input_embedding = self.code_embedding(decoder_input)
                decoder_input_sketch_embedding = decoder_input_sketch_embedding.unsqueeze(1)
                decoder_input_embedding = decoder_input_embedding.unsqueeze(1)
                code_pred_logits.append(cur_code_pred_logits)
                code_predictions.append(cur_code_predictions)
                cur_predictions = cur_code_predictions
                if self.hierarchy:
                    df_pred_logits.append(cur_df_pred_logits)
                    df_predictions.append(cur_df_predictions)
                    var_pred_logits.append(cur_var_pred_logits)
                    var_predictions.append(cur_var_predictions)
                    str_pred_logits.append(cur_str_pred_logits)
                    str_predictions.append(cur_str_predictions)
                    cur_predictions = torch.max(cur_predictions, cur_df_predictions)
                    cur_predictions = torch.max(cur_predictions, cur_var_predictions)
                    cur_predictions = torch.max(cur_predictions, cur_str_predictions)
                predictions.append(cur_predictions)

                cur_finished = (decoder_input == data_utils.EOS_ID).long().unsqueeze(1)
                finished = torch.max(finished, cur_finished)
                '''
                if torch.sum(finished) == batch_size:
                    break
                '''

            total_loss = 0.0
            code_pred_logits = torch.stack(code_pred_logits, dim=0)
            code_pred_logits = code_pred_logits.permute(1, 2, 0)
            code_predictions = torch.stack(code_predictions, dim=0)
            code_predictions = code_predictions.permute(1, 0)
            total_loss += F.cross_entropy(code_pred_logits, target_code_output, ignore_index=data_utils.PAD_ID)

            if self.label_prediction:
                total_loss += F.cross_entropy(plot_type_classification_logits, batch_labels)
            if self.hierarchy:
                df_pred_logits = torch.stack(df_pred_logits, dim=0)
                df_pred_logits = df_pred_logits.permute(1, 2, 0)
                df_predictions = torch.stack(df_predictions, dim=0)
                df_predictions = df_predictions.permute(1, 0)
                df_loss = F.cross_entropy(df_pred_logits, target_df_output, ignore_index=-1)

                var_pred_logits = torch.stack(var_pred_logits, dim=0)
                var_pred_logits = var_pred_logits.permute(1, 2, 0)
                var_predictions = torch.stack(var_predictions, dim=0)
                var_predictions = var_predictions.permute(1, 0)
                var_loss = F.cross_entropy(var_pred_logits, target_var_output, ignore_index=-1)

                str_pred_logits = torch.stack(str_pred_logits, dim=0)
                str_pred_logits = str_pred_logits.permute(1, 2, 0)
                str_predictions = torch.stack(str_predictions, dim=0)
                str_predictions = str_predictions.permute(1, 0)
                str_loss = F.cross_entropy(str_pred_logits, target_str_output, ignore_index=-1)
                total_loss += (df_loss + var_loss + str_loss) / 3.0

            predictions = torch.stack(predictions, dim=0)
            predictions = predictions.permute(1, 0)
            return total_loss, code_pred_logits, predictions
