# -*- coding: utf-8 -*-
"""
=============================================================================================================================
        Gaussian Softmax Model ( ICML 2017 Discovering Discrete Latent Topics with Neural Variational Inference)
        Implemented by lly_aegis@foxmail.com
        Version 1.001  2018 - 06 - 12
        Release note
        1.001 Optimized the memory allocation efficiency for dataset
=============================================================================================================================
"""
from __future__ import print_function
import argparse, sys, os
import torch
import torch.utils.data
# import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
#====================================================================================================================================================
parser = argparse.ArgumentParser(description='Gaussian Softmax Model parameters description.')
parser.add_argument('--hidden', type=int, default=256, metavar='N', 
    help="The size of hidden units in MLP inference network (default 256)")
parser.add_argument('--dropout', type=float, default=0.8, metavar='N', 
    help="The drop-out probability of MLP (default 0.8)")
parser.add_argument('--lr', type=float, default=1e-5, metavar='N', 
    help="The learning rate of model (default 1e-5)")
parser.add_argument('--topics', type=int, default=50, metavar='N',
    help="The amount of topics to be discover (default 50)")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    help="Training batch size.")
parser.add_argument('--vocab-size', type=int, default=2000, metavar='N',
    help="Vocabulary size of topic modelling")
parser.add_argument('--topicembedsize', type=int, default=128, metavar='N',
    help="Topic embedding size of topic modelling")
parser.add_argument('--alternative-epoch', type=int, default=10, metavar='N',
    help="Alternative epoch size for wake sleep algorithm")
parser.add_argument('--training-epoch', type=int, default=1000, metavar='N',
    help="Alternative epoch size for wake sleep algorithm")
parser.add_argument('--cuda', action='store_true', default=True,
    help="Flag for disable CUDA training.")
parser.add_argument('--inf-nonlinearity', default='tanh', metavar='N',
    help="Options for non-linear function.(default tanh)")
parser.add_argument('--data-path', default='data/20news/', metavar='N',
    help="Directory for corpus")
parser.add_argument('--word-vector-path', default='/home/tyk/glove/GloVe-1.2/20news/vectors.txt', metavar='N',
    help="Directory for word vector")
parser.add_argument('--gamma', type=float, default=0.001, metavar='N',
    help="beta")
parser.add_argument('--model-save-path', default='new_sav_gmm_drop3_topic200gamma10-2/', metavar='N',
    help="Directory for corpus")
#=================================================Global parameter settings========================================================================
# MODEL_SAV_PATH='r1_glove_sav_drop3_topic20vec512gamm10-2/'
MAX_TO_KEEP=5
TOPN=20
#========================================================================================================================
torch.manual_seed(0)
args=parser.parse_args()
MODEL_SAV_PATH=args.model_save_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#====================================================================================================================================================
class Inference_network(nn.Module):
    def __init__(self, vocabulary_size=2000, n_hidden=256, dropout_prob=0.8, n_topics=50, inf_nonlinearity='tanh'):
        super(Inference_network, self).__init__()
        '''
        if inf_nonlinearity == 'sigmoid':
            self.inf_nonlinearity=nn.Sigmoid
        elif inf_nonlinearity == 'relu':
            self.inf_nonlinearity=nn.ReLu
        else: self.inf_nonlinearity = nn.Tanh'''
        self.n_topics=n_topics
        self.dropout=nn.Dropout(dropout_prob)
        self.MLP_act=nn.Tanh()
        self.linear=nn.Linear(vocabulary_size, n_hidden)
        torch.nn.init.xavier_uniform_(self.linear.weight.data, gain=1)
        #self.linear.bias.data.fill_(0.0)
        self.mu=nn.Linear(n_hidden, n_topics)
        #self.mu_norm = torch.nn.BatchNorm1d(n_topics)
        self.logsig=nn.Linear(n_hidden, n_topics)
        #self.logsig_norm = torch.nn.BatchNorm1d(n_topics)
        #Weight and bias zero initialization
        self.logsig.weight.data.fill_(0.0)
        self.logsig.bias.data.fill_(0.0)
        torch.nn.init.xavier_uniform_(self.mu.weight.data, gain=1)
        #self.mu.bias.data.fill_(0.0)
        #self.prior_Gaussian = torch.distributions.normal.Normal(loc= torch.zeros(self.n_topics).to(device), scale = torch.ones(self.n_topics).to(device))

    def forward(self, doc_bow):
        enc_vec=self.dropout(self.MLP_act(self.linear(doc_bow)))
        mu = self.mu(enc_vec)
        logsig = self.logsig(enc_vec)
        #mu_norm=self.mu_norm(self.mu(enc_vec))
        #logsig_norm=self.logsig_norm(self.logsig(enc_vec))
        #encoded_Gaussian = torch.distributions.normal.Normal(loc = mu, scale = torch.exp(2*logsig))
        #KL = torch.distributions.kl.kl_divergence(encoded_Gaussian, self.prior_Gaussian)
        KL= -0.5 * torch.sum(1 - torch.pow(mu ,2) + 2 * logsig - torch.exp(2*logsig), 1)
        return mu, logsig, KL 
#---------------------------------------------------------------------------------------------------------------------------------------------------
class Generative_model(nn.Module):
    def __init__(self, vocabulary_size=2000, n_topics=50, topic_embeddings_size=128,word_embedding=torch.zeros([1, 1], dtype=torch.float64)):
        super(Generative_model, self).__init__() 
        self.n_topics = n_topics
        self.vocabulary_size = vocabulary_size
        self.topic_embeddings_size = topic_embeddings_size
        
        self.theta_softmax=nn.Softmax(dim=0)
        #Initializing linear layer weight and bias
        self.theta_linear=nn.Linear(n_topics, n_topics)
        #torch.nn.init.xavier_uniform(self.theta_linear.weight.data, gain=1)
        self.theta_linear.bias.data.fill_(0.0)
        self.theta_linear.weight.data.fill_(0.0)

        self.beta_softmax = nn.Softmax(dim=1)
        #========================================================================================
        # topic_embeddings_mat & word_embeddings_mat didnt registered as model parameter
        #========================================================================================
        topic_embeddings_mat = Parameter(torch.Tensor(n_topics, topic_embeddings_size))
        torch.nn.init.xavier_uniform_(topic_embeddings_mat.data, gain=1)
        self.register_parameter('topic_embeddings_mat', topic_embeddings_mat)
        # print(word_embedding)
        # exit()
        self.word_embeddings_mat = word_embedding.to(device)
    
        mu_mat = Parameter(torch.Tensor(n_topics, topic_embeddings_size))
        torch.nn.init.xavier_normal_(mu_mat.data, gain=1)
        # torch.nn.init.kaiming_uniform_(mu_mat.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.register_parameter('mu_mat', mu_mat)

        log_sigma2_mat = Parameter(torch.Tensor(n_topics, topic_embeddings_size))
        # torch.nn.init.xavier_normal_(log_sigma2_mat.data)
        torch.nn.init.xavier_normal_(log_sigma2_mat.data, gain=1)
        # torch.nn.init.xavier_normal_(log_sigma2_mat.data, gain=1)
        # torch.nn.init.kaiming_uniform_(log_sigma2_mat.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.register_parameter('log_sigma2_mat', log_sigma2_mat)
    def reparameterization(self, mu, logsig):
        std = torch.exp(logsig)
        eps= torch.normal(mean = torch.zeros(self.n_topics)).to(device)
        return torch.mul(eps, std).add_(mu).to(device)

    def log_pdf(self, mu, logsig,vec):
        G=[]
        # print(mu.size())
        # print(logsig.size())
        # print(vec.size())
        # print(mu)
        # print(logsig)
        for k in range(self.n_topics):
            pdf=-0.5*torch.sum(math.log(math.pi*2)+logsig[k:k+1,:]+(vec-mu[k:k+1,:]).pow(2)/torch.exp(logsig[k:k+1,:]),1).view(1,-1)
            # print(pdf)
            G.append(pdf)
        return torch.cat(G,0)

    #Working flow of generative model
    def forward(self, mu, logsig, doc_bow):
        delta=1e-10
        z = self.reparameterization(mu, logsig)
        # print(z)
        self.theta = self.theta_softmax(self.theta_linear(z))
        
        self.gaussian_pdf = self.log_pdf(self.mu_mat, self.log_sigma2_mat,self.word_embeddings_mat)
        # print('self.gaussian_pdf')
        # print (self.gaussian_pdf)
        self.beta_ori = self.beta_softmax(self.gaussian_pdf)
        # print(torch.kthvalue(self.beta_ori,10))

    

        self.beta = (self.beta_ori + delta) / (torch.sum(self.beta_ori,1).view(-1,1)+delta*self.beta_ori.size()[1])
     
        logits = torch.log(torch.mm(self.theta, self.beta)+1e-10)
        # print('logits')
        # print(logits)
        # print('------------------------------')
        Re= - torch.sum(torch.mul(logits, doc_bow), 1)
        # print(Re)
        return Re, self.theta, self.beta
#====================================================================================================================================================
class NVGMTM(nn.Module):
    def __init__(self, vocabulary_size=2000, n_hidden=256, dropout_prob=0.8, n_topics=50, topic_embeddings_size=128, inf_nonlinearity='tanh', alternative_epoch=10,gamma=0.001,word_embedding=torch.zeros([1, 1], dtype=torch.float64)):
        super(NVGMTM, self).__init__()
        #Inference network component
        self.gamma = gamma
        self.n_topics = n_topics
        self.inf_net=Inference_network(vocabulary_size, n_hidden, dropout_prob, n_topics, inf_nonlinearity)
        #Generative model component
        self.gen_model=Generative_model(vocabulary_size, n_topics, topic_embeddings_size,word_embedding)
    def forward(self, doc_bow):
        #Invoke Inference network working flow
        # print('self.gama',self.gama)
        mu, logsigm, KL = self.inf_net(doc_bow)
        #Invoke Generative model working flow
        Re, theta, gaussian_pdf = self.gen_model(mu, logsigm, doc_bow)
        loss = Re + KL  
        # new_loss = loss - self.gamma*torch.sum(torch.sum(gaussian_pdf,1))/doc_bow.size(0)
        new_loss = loss - self.gamma*torch.sum(torch.sum(torch.log(gaussian_pdf),1))/doc_bow.size(0)
        # print(Re)
        # print(KL)
        # print(loss*doc_bow.size(0))
        # print(torch.mean(torch.sum(gaussian_pdf,1)))
        # exit()
        # print(torch.sum(torch.sum(gaussian_pdf,1),0))
        # exit()
        # new_loss = Re + KL + GMMLoss
        return loss, KL, theta, new_loss
    def enable_inf_net_training(self):
        for param in self.inf_net.parameters():
            param.requires_grad = True
        for param in self.gen_model.parameters():
            param.requires_grad = False
    def enable_gen_model_training(self):
        for param in self.inf_net.parameters():
            param.requires_grad = False
        for param in self.gen_model.parameters():
            param.requires_grad = True
#------------------------------------------------------------------------------------------------------------------------
    def train_and_eval(self, 
                       train_dataloader, 
                       test_dataloader, 
                       learning_rate=1e-4, 
                       batch_size=65, 
                       training_epoch=1000, 
                       alternative_epoch=10):
        inf_optim=optim.Adam(self.inf_net.parameters(), lr=learning_rate)
        gen_optim=optim.Adam(self.gen_model.parameters(), lr=learning_rate)
        # gmm_encoder_optim=optim.Adam(self.gmmencoder.parameters(), lr=learning_rate)
        # gmm_decoder_optim=optim.Adam(self.gmmdecoder.parameters(), lr=learning_rate)
        min_ppx=9999.0
        test_ppx_trend=[]
        test_kld_trend=[]
        no_decent_cnt = 0
        word2id, id2word, vocabulary = ReadDictionary(args.data_path + 'vocab.new')
        # If previous checkpoint files exist, load the pretrain paramter dictionary from them.
        ckpt = 0
        if os.path.exists(MODEL_SAV_PATH):
            ckpt_list = os.listdir(MODEL_SAV_PATH)
            if len(ckpt_list)>0:
                for ckpt_f in ckpt_list:
                    current_ckpt = int(ckpt_f.split('-')[1].split('.')[0])
                    if current_ckpt > ckpt:
                        ckpt = current_ckpt
                self.load_state_dict(torch.load(MODEL_SAV_PATH + "model_parameters_epoch-"+str(ckpt)+".pkl"))
        else:
            os.makedirs(MODEL_SAV_PATH)
        #------------------------------------------------------------------------------------
        #Main training epoch control
        for epoch in range(ckpt, training_epoch):
            self.train()
            for mode, optimizer in enumerate([gen_optim, inf_optim]):
                if mode==1:
                    optim_mode="Updating Encoder parameters "
                    self.enable_inf_net_training()
                else:
                    optim_mode="Updating Decoder parameters "
                    self.enable_gen_model_training()
                #Alternative training control for wake-sleep algorithm
                for sub_epoch in range(alternative_epoch):
                    loss_sum=0.0
                    ppl_sum=0.0
                    kld_sum=0.0
                    training_word_count=0
                    doc_count=0
                    new_loss_sum=0.0
                    for batch_idx, (data, lable, word_count) in enumerate(train_dataloader):
                        word_count = word_count.float().to(device)
                        data_size=len(data)
                        data = data.to(device)
                        word_count=word_count.to(device)
                        optimizer.zero_grad()
                        loss, KL, theta, new_loss=self(data)
                        # new_loss/=data_size
                        new_loss.backward(torch.FloatTensor(torch.ones(data_size)).to(device))
                        optimizer.step()
                        #Computing evaluation metrics
                        new_loss_sum += torch.sum(new_loss)
                        loss_sum += torch.sum(loss)
                        kld_sum += torch.sum(KL) / data_size
                        training_word_count += torch.sum(word_count)
                        ppl_sum += torch.sum(torch.div(loss, word_count))
                        doc_count += len(data)
                    corpus_ppl = torch.exp(loss_sum / training_word_count)
                    #.data.to("cpu")
                    per_doc_ppl= torch.exp(ppl_sum /doc_count)
                    #.data.to("cpu")
                    # new_loss_sum=new_loss_sum/doc_count
                    kldc=torch.div(kld_sum, len(train_dataloader))
                    #.to("cpu")
                    print('| %s | Training epoch %2d | Corpus PPX: %.5f | Per doc PPX: %.5f | KLD: %.5f | loss: %.5f' % (optim_mode, 
                        sub_epoch+1, 
                        corpus_ppl, 
                        per_doc_ppl, 
                        kldc,new_loss_sum))
                    # exit()
            #==========================================================================================================
            #Evaluating model on testset
            self.eval()
            loss_sum=0.0
            ppl_sum=0.0
            kld_sum=0.0
            new_loss_sum=0.0
            training_word_count=0
            doc_count=0
            for batch_idx, (data, lable, word_count) in enumerate(test_dataloader):
                data=data.to(device)
                word_count = word_count.float().to(device)
                loss, KL, theta, new_loss = self(data)
                new_loss_sum += torch.sum(new_loss)# / len(data)
                loss_sum += torch.sum(loss)
                kld_sum += torch.sum(KL) / len(data)
                training_word_count += torch.sum(word_count)
                ppl_sum += torch.sum(torch.div(loss, word_count))
                doc_count += len(data)
            test_ppl = torch.exp(loss_sum / training_word_count)
            #.data.to("cpu")
            per_doc_ppl= torch.exp(ppl_sum / doc_count)
            #.data.to("cpu")
            # new_loss_sum = new_loss_sum/doc_count#, len(test_dataloader))
            kldc = torch.div(kld_sum, len(test_dataloader))
            #.data.to("cpu")
            print('===Test epoch %2d ===| Testset PPX: %.5f | Per doc PPX: %.5f | KLD: %.5f | loss: %.5f' % (epoch+1, 
                test_ppl, 
                per_doc_ppl, 
                kldc,new_loss_sum))
            #Recording training statics
            test_kld_trend.append(kldc.detach().cpu().numpy())
            curr_test_ppl = float(test_ppl.detach().cpu().numpy())
            test_ppx_trend.append(curr_test_ppl)
            #============================================================================================
            # Export the current topic-word distribution file (topn indicates the numbers of top-n words retrieved from beta matrix )
            # Serialize the best performance model parameters
            if epoch+1 > 1:
                #Implementation for exporting topic-word distribution files.
                mat_beta = self.gen_model.beta.detach().cpu().numpy()
                mat_theta = self.gen_model.theta.detach().cpu().numpy()
                mat_log_sigma2 = self.gen_model.log_sigma2_mat.detach().cpu().numpy()
                mat_mu = self.gen_model.mu_mat.detach().cpu().numpy()
                # mat_word = self.gen_model.word_embeddings_mat.detach().cpu().numpy()
                topic_coherence_file_export(mat_beta, id2word, TOPN, mat_theta, mat_log_sigma2, mat_mu)
                #TODO: Serialize better model and Early stop when not consecutively decent for 30 epoch
                if min_ppx > curr_test_ppl:
                    no_decent_cnt = 0
                    min_ppx = curr_test_ppl
                    #Save all model parameters to designated path.
                    torch.save(self.state_dict(), MODEL_SAV_PATH + "model_parameters_epoch-"+str(epoch+1)+".pkl")
                    #Max_to_keep mechanism implementation
                    ckpt_tmp_list = os.listdir(MODEL_SAV_PATH)
                    if len(ckpt_tmp_list) > MAX_TO_KEEP :
                        os.remove(MODEL_SAV_PATH + ckpt_tmp_list[0])
                else:
                    #Early-stop
                    no_decent_cnt+=1
                    if no_decent_cnt > 30:
                        break
        
        return test_ppx_trend, test_kld_trend
            
#====================================================================================================================================================
#====================================================================================================================================================
#Corpus object
class BOW_TopicModel_Corpus(torch.utils.data.Dataset):
    def __init__(self, vocabulary_size, data_path, loader=None):
        #data_path: path route for train.feat and test.feat
        self.vocabulary_size=vocabulary_size
        self.doc_set = []
        self.doc_count=0.0
        doc_index=0

        with open(data_path, 'r') as f:
            docs=f.readlines()
            for doc in docs:
                self.doc_set.append(doc)
                doc_index+=1
        self.loader=loader
        self.doc_count=doc_index

    def __len__(self):
        return len(self.doc_set)
    
    def tokens2vec(self, token_list, vocabulary_size):
        vec=torch.zeros(vocabulary_size)
        word_count=0
        for token in token_list:
            # <token index>:tf
            token_index=int(token.split(':')[0])
            token_tf=int(token.split(':')[1])
            word_count+=token_tf
            vec[token_index-1]=token_tf
        return vec, word_count

    def __getitem__(self, index):
        item_list=self.doc_set[index].strip().split(' ')
        class_label=item_list[0]
        doc_tokens=item_list[1:]
        vec, word_count = self.tokens2vec(doc_tokens, self.vocabulary_size)
        #return self.doc_vec_set[index], self.lable_set[index], self.word_count_set[index]
        return vec, class_label, word_count
#==============================================================================================================
def ReadWordVector(vectorpath):
    fp=open(vectorpath,'r').readlines()
    vectors=[]
    # i=0
    # for v in fp[:-1]:
    for v in fp:
        v=v.strip().split()
        # vv = [float(i) for i in v[1:]]
        vectors.append([float(i) for i in v])
        # print(v[1:])
        # exit()
        # print(i)
        # i+=1
    # print('done')
    # vectors=np.transpose(vectors)
    # print('np.transpose(vectors)')
    # print(len(vectors),len(vectors[0]))
    vectors=torch.Tensor(vectors)
    # print('torch.from_numpy(vectors)')
    # vectors=vectors.t()
    # print('vectors.to(device).t()')
    return vectors

def ReadDoc(name):
    fp=open(name,'r')
    doc=fp.readlines()
    fp.close()
    return doc

def ReadDictionary(vocabpath):
    word2id=dict()
    id2word=dict()
    vocabulary=dict()
    txt=ReadDoc(vocabpath)
    for i in range(0,len(txt)):
        if len(txt)>2:
            tmp_list=txt[i].strip().split(' ')
            word2id[tmp_list[0]]=i
            id2word[i]=tmp_list[0]
            vocabulary[tmp_list[0]]=tmp_list[1]
    return word2id, id2word, vocabulary

def topic_coherence_file_export(topicmat, id2word, topn, mat_theta, mat_log_sigma2, mat_mu):
    #Input topicmat need to be a numpy ndarray
    #Read vocabulary dict from corpus. data/20news/vocab.new or data/rcv1-v2/vocab.new
    matrix_path = MODEL_SAV_PATH[0:-1] +"matrix/"
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path) 
    f=open(matrix_path +"topics_gmm.txt", 'w')
    # ftopicmat=open(MODEL_SAV_PATH +"beta.txt", 'w')
    # fmat_theta=open(MODEL_SAV_PATH +"theta.txt", 'w')
    # fmat_topic=open(MODEL_SAV_PATH +"topic_embedding.txt", 'w')
    # fmat_word=open(MODEL_SAV_PATH +"word_embedding.txt", 'w')  
    np.save(matrix_path+"beta.txt", topicmat)
    np.save(matrix_path+"theta.txt", mat_theta)
    np.save(matrix_path+"logsigma_embedding.txt", mat_log_sigma2)
    np.save(matrix_path+"mu_embedding.txt", mat_mu)
    # np.save(matrix_path+"word_embedding.txt", mat_word)
    # f=open("topics_gmm.txt", 'w')
    for topic in topicmat:
        topics_word_list=[]
        word_cnt=1
        tmp_list=[]
        # Build word, probability tuple for each topic
        for index, value in enumerate(topic):
            tmp_list.append((index, value))
        #Decently sort the word according to its probability
        sorted_list = sorted(tmp_list, key = lambda s:s[:][1], reverse=True)
        for pair in sorted_list:
            if word_cnt > topn: 
                break
            if pair[0] not in id2word.keys():
                print(pair[0])
                sys.exit(0)
            topics_word_list.append(id2word[pair[0]])
            word_cnt+=1
        f.write(' '.join(topics_word_list)+'\n')
    f.close()

#==============================================================================================================
#==============================================================================================================
def main(args):
    #Load dataset
    print('Hidden units: %d , drop-out rate: %f , topic number: %d' % (args.hidden, args.dropout, args.topics))
    train_dataset=BOW_TopicModel_Corpus(vocabulary_size=args.vocab_size, data_path=args.data_path+'train.feat')
    test_dataset=BOW_TopicModel_Corpus(vocabulary_size=args.vocab_size, data_path=args.data_path +'test.feat')
    #Build dataloader
    word_vectors=ReadWordVector(args.word_vector_path)
    # print(word_vectors)
    # print('read vector done!!!!!!!!!!!!!!!!!!!!')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        # num_workers=1,
        num_workers=0, 
        pin_memory=True, 
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        # num_workers=1,
        pin_memory=True,      
        drop_last=False)
    #Build model
    model = NVGMTM(vocabulary_size=args.vocab_size, 
        n_hidden=args.hidden, 
        dropout_prob=args.dropout, 
        n_topics=args.topics, 
        topic_embeddings_size=args.topicembedsize, 
        inf_nonlinearity=args.inf_nonlinearity, 
        alternative_epoch=args.alternative_epoch,
        gamma=args.gamma,
        word_embedding=word_vectors)
    model=model.to(device)
    #Train & test process
    test_ppx_trend, test_kld_trend = model.train_and_eval(train_loader, test_loader, batch_size=args.batch_size, learning_rate=1e-4)
    #TODO: Plot test ppx trend and kld trend

    #TODO: Export the topic word distribution matrix when finished training
    
if __name__ == '__main__':
    main(args)


