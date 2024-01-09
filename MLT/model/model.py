import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
import time
ISOTIMEFORMAT='%Y-%m-%d %X'

class JointModel(nn.Module):
    def __init__(self, d_v, d_e, d_t, encoder_layers, generator_layers,encoder_shortcut, generator_shortcut, generator_transform,
                 num_word, emb_size, word_rnn_size, word_rnn_num_layer, word_rnn_dropout, word_rnn_bidirectional,word_attention_size,
                 context_rnn_size, context_rnn_num_layer, context_rnn_dropout, context_rnn_bidirectional,context_attention_size, mlp_size,
                 num_label, pretrained_embedding, vocab):

        super(JointModel, self).__init__()

        ##NGTM:
        self.d_v = d_v  # vocabulary size
        self.d_e = d_e  # dimensionality of encoder
        self.d_t = d_t  # number of topics
        self.encoder_layers = encoder_layers
        self.generator_layers = generator_layers
        self.generator_transform = generator_transform  # transform to apply after the generator
        self.encoder_shortcut = encoder_shortcut
        self.generator_shortcut = generator_shortcut        
        self.en1_fc = nn.Linear(self.d_v, self.d_e)
        self.en2_fc = nn.Linear(self.d_e, self.d_e)
        self.en_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(self.d_e, self.d_t)
        #         self.mean_bn = nn.BatchNorm1d(self.d_t)
        self.logvar_fc = nn.Linear(self.d_e, self.d_t)
        #         self.logvar_bn = nn.BatchNorm1d(self.d_t)
        self.generator1 = nn.Linear(self.d_t, self.d_t)
        self.generator2 = nn.Linear(self.d_t, self.d_t)
        self.generator3 = nn.Linear(self.d_t, self.d_t)
        self.generator4 = nn.Linear(self.d_t, self.d_t)
        self.r_drop = nn.Dropout(0.2)
        self.de = nn.Linear(self.d_t, self.d_v)
        self.vocab = vocab
        #         self.de_bn = nn.BatchNorm1d(self.d_v)
    
        ##HAN:
        self.emb_size = emb_size
        self.word_rnn_size = word_rnn_size
        self.word_rnn_num_layer = word_rnn_num_layer
        self.word_rnn_bidirectional = word_rnn_bidirectional
        self.context_rnn_size = context_rnn_size
        self.context_rnn_num_layer = context_rnn_num_layer
        self.context_rnn_bidirectional = context_rnn_bidirectional
        self.num_label = num_label
        self.embedding = nn.Embedding(num_word, emb_size)
        self.word_rnn = nn.GRU(input_size=emb_size, hidden_size=word_rnn_size, dropout=word_rnn_dropout,
                               num_layers=word_rnn_num_layer, bidirectional=word_rnn_bidirectional)
        word_rnn_output_size = word_rnn_size * 2 if word_rnn_bidirectional else word_rnn_size
        self.word_conv_attention_linear = nn.Linear(word_rnn_output_size, self.d_t, bias=False)
        self.word_conv_attention_linear2 = nn.Linear(self.d_t, 1, bias=False)
        self.context_rnn = nn.GRU(input_size=word_rnn_output_size, hidden_size=context_rnn_size,dropout=context_rnn_dropout,
                                  num_layers=context_rnn_num_layer, bidirectional=context_rnn_bidirectional)
        context_rnn_output_size = context_rnn_size * 2 if context_rnn_bidirectional else context_rnn_size
        self.context_conv_attention_linear = nn.Linear(context_rnn_output_size, 1, bias=False)
        self.classifier = nn.Sequential(nn.Linear(context_rnn_output_size, mlp_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(mlp_size, num_label),
                                       nn.Tanh())
        '''       
        ###BERT 
        # Inizializza il tokenizzatore BERT
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Aggiungi il layer BERT pre-addestrato
        #self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        custom_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_state=True)
        custom_config.max_position_embeddings = 3000
        self.bert = BertModel(config=custom_config)
        # Definisci il resto della tua struttura, adattando l'input al tuo modello
        self.classifier = nn.Sequential(nn.Linear(3000, mlp_size), nn.LeakyReLU(), nn.Linear(mlp_size, num_label), nn.Tanh())
        '''
    def convert_ids2words(self, id_sequences, vocab):
        inverted_vocab = {v: k for k, v in vocab.items()}  # Inverti il dizionario
        return [[inverted_vocab.get(id, "<unk>") for id in seq] for seq in id_sequences]

       # if pretrained_embedding is not None:
            #self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)


    def encoder(self, x):
        if self.encoder_layers == 1:
            pi = F.relu(self.en1_fc(x))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)
        else:
            pi = F.relu(self.en1_fc(x))
            pi = F.relu(self.en2_fc(pi))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)

        #         mean = self.mean_bn(self.mean_fc(pi))
        #         logvar = self.logvar_bn(self.logvar_fc(pi))
        mean = self.mean_fc(pi)
        logvar = self.logvar_fc(pi)
        return mean, logvar

    def sampler(self, mean, logvar, cuda):
        eps = torch.randn(mean.size())
        sigma = torch.exp(logvar)
        h = sigma.mul(eps).add_(mean)
        return h

    def generator(self, h):
        if self.generator_layers == 0:
            r = h
        elif self.generator_layers == 1:
            temp = self.generator1(h)
            if self.generator_shortcut:
                r = torch.tanh(temp) + h
            else:
                r = temp
        elif self.generator_layers == 2:
            temp = torch.tanh(self.generator1(h))
            temp2 = self.generator2(temp)
            if self.generator_shortcut:
                r = torch.tanh(temp2) + h
            else:
                r = temp2
        else:
            temp = torch.tanh(self.generator1(h))
            temp2 = torch.tanh(self.generator2(temp))
            temp3 = torch.tanh(self.generator3(temp2))
            temp4 = self.generator4(temp3)
            if self.generator_shortcut:
                r = torch.tanh(temp4) + h
            else:
                r = temp4

        if self.generator_transform == 'tanh':
            return self.r_drop(torch.tanh(r))
        elif self.generator_transform == 'softmax':
            return self.r_drop(F.softmax(r,dim=1)[0])
        elif self.generator_transform == 'relu':
            return self.r_drop(F.relu(r))
        else:
            return self.r_drop(r)

    def decoder(self, r):
        #         p_x_given_h = F.softmax(self.de_bn(self.de(r)),dim=1)
        p_x_given_h = F.softmax(self.de(r),dim=1)
        return p_x_given_h

    def init_rnn_hidden(self, batch_size, level):
        param_data = next(self.parameters()).data
        if level == "word":
            bidirectional_multipier = 2 if self.word_rnn_bidirectional else 1
            layer_size = self.word_rnn_num_layer * bidirectional_multipier
            word_rnn_init_hidden = param_data.new(layer_size, batch_size, self.word_rnn_size).zero_()
            return word_rnn_init_hidden
        elif level == "context":
            bidirectional_multipier = 2 if self.context_rnn_bidirectional else 1
            layer_size = self.context_rnn_num_layer * bidirectional_multipier
            context_rnn_init_hidden = param_data.new(layer_size, batch_size, self.context_rnn_size).zero_()
            return context_rnn_init_hidden
        else:
            raise Exception("level must be 'word' or 'context'")

    def continuous_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("selector"):
                yield param

    def discrete_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("selector"):
                yield param

    def forward(self, x, x_indices, input_list, length_list, cuda, vocab):
        '''
        print("vae start")
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        '''
        ###topic model
        mean, logvar = self.encoder(x)# batchsize*50
        h = self.sampler(mean, logvar, cuda)# batchsize*50
        #to apply the dirichlet distribution we need to apply a softmax layer to the distribution h in order to obtain a distribution that follows the dirichlet
        h_dirc = F.softmax(h,dim=1)#now we have h that follow dirchlet distribution
        r = self.generator(h_dirc)# batchsize*50
        p_x_given_h = self.decoder(r) # batchsize*dv
        ###HAN
        word_attention_dict = {}
        
        num_utterance = len(input_list)  # one batch doucument_list
        _, batch_size = input_list[0].size()
        
        # word-level rnn
        word_rnn_hidden = self.init_rnn_hidden(batch_size, level="word")
        word_rnn_output_list = []
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_output, word_rnn_hidden = self.word_rnn(word_rnn_input, word_rnn_hidden)
            word_attention_weight = self.word_conv_attention_linear(word_rnn_output)

            batch_data = input_list[utterance_index]
            for word_i in range(len(batch_data)):  # word_i word
                for clause_i in range(len(batch_data[word_i])):  # clause_i data（batch）
                    word_index = int(batch_data[word_i, clause_i])  # word index
                    if word_index < self.d_v:
                        if word_index in word_attention_dict:
                            word_attention_dict[word_index] = (word_attention_dict[word_index] + word_attention_weight[word_i, clause_i,:]) / 2
                        else:
                            word_attention_dict[word_index] = word_attention_weight[word_i, clause_i, :]

            ##HAN
            word_attention_weight = self.word_conv_attention_linear2(word_attention_weight)
            word_attention_weight = nn.functional.relu(word_attention_weight)
            word_attention_weight = nn.functional.softmax(word_attention_weight, dim=0)
            word_rnn_last_output = torch.mul(word_rnn_output, word_attention_weight).sum(dim=0)
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()
            # context-level rnn
        context_rnn_hidden = self.init_rnn_hidden(batch_size, level="context")
        context_rnn_input = torch.stack(word_rnn_output_list, dim=0)
        context_rnn_output, context_rnn_hidden = self.context_rnn(context_rnn_input, context_rnn_hidden)
        context_attention_weight = self.context_conv_attention_linear(context_rnn_output)
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = nn.functional.softmax(context_attention_weight, dim=0)
        context_rnn_last_output = torch.mul(context_rnn_output, context_attention_weight).sum(dim=0)        
        classifier_input = context_rnn_last_output
        logit = self.classifier(classifier_input)
        '''
        max_chunk_length = 64
        # Converte le rappresentazioni numeriche in testo utilizzando il tuo dizionario inverso
        tensor_inside_list = input_list[0]  # Assumendo che il tensore sia nel primo elemento della lista.

        # Converti il tensore in una lista di liste.
        list_of_lists = tensor_inside_list.tolist()

        # Ora puoi chiamare la tua funzione convert_ids2words.
        testo = self.convert_ids2words(list_of_lists, self.vocab)
        text_list = [' '.join(sentence) for sentence in testo]
        # Utilizza il tokenizer BERT per ottenere gli ID delle parole dai testi
        tokenized_inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=max_chunk_length)
        # Calcola il numero di chunk che vuoi dividere
        num_chunks = tokenized_inputs['input_ids'].size(1) // max_chunk_length

        # Inizializza una lista vuota per contenere i chunk
        input_id_chunks = []

        # Dividi il tensore input_ids in chunk e aggiungili alla lista
        for i in range(num_chunks):
            start = i * max_chunk_length
            end = start + max_chunk_length
            chunk = tokenized_inputs['input_ids'][:, start:end]
            input_id_chunks.append(chunk)
        # Assicurati che ogni chunk abbia [CLS] all'inizio e [SEP] alla fine
        # ... (codice precedente)
        start_token = torch.tensor([101]).view(-1, 1)  # Inizializza il token di inizio con dimensione 1 x 1
        end_token = torch.tensor([102]).view(-1, 1)    # Inizializza il token di fine con dimensione 1 x 1
        input_id_chunks = [torch.cat([start_token.expand(chunk.size(0), -1), chunk, end_token.expand(chunk.size(0), -1)], dim=1) for chunk in input_id_chunks]
        # Ora chunks contiene la lista di chunk con lunghezza massima max_chunk_length o men
        bert_hidden_states_list = []
        for chunk in input_id_chunks:
            input_dict = {'input_ids': chunk,}
            bert_output = self.bert(**input_dict)
            if bert_output is not None and bert_output.hidden_states is not None and len(bert_output.hidden_states) > 0:
                print("the hidden states are:")
                print(bert_output.hidden_states)
                bert_hidden_states_list.append(bert_output.hidden_states)
        if len(bert_hidden_states_list) > 0:
            # Combina le rappresentazioni nascoste dei chunks
            combined_bert_hidden_states = torch.cat(bert_hidden_states_list, dim=1)
        else:
            # Nessuna rappresentazione nascosta valida disponibile, puoi eseguire un fallback o gestire l'assenza di input
            #print("no hidden representation with bert ")
            combined_bert_hidden_states = torch.zeros_like(x)
     # Utilizza le rappresentazioni nascoste di BERT come input al resto del tuo modello
        logit = self.classifier(combined_bert_hidden_states)
        '''
        return mean, logvar, p_x_given_h, logit, word_attention_dict 
