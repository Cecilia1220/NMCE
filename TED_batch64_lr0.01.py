import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import sklearn.datasets as datasets

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from torch.utils.data import Dataset, DataLoader
from loss import MaximalCodingRateReduction, Z_loss, TotalCodingRate
from func import chunk_avg, cluster_match, analyze_latent, save_cluster_imgs,cluster_acc
from architectures.models import Gumble_Softmax, SubspaceClusterNetwork
from data.aug import GaussianBlur
from evals.cluster import kmeans, ensc

from transformers import BertModel, BertTokenizer 

# Check GPU
print('Torch Cuda Is Available: ', torch.cuda.is_available())

print('Torch Cuda Device Count: ', torch.cuda.device_count())

print('Torch Current Device: ', torch.cuda.current_device())

print('Torch Cuda Device #: ', torch.cuda.device(0))

print('Torch Cuda Device Name: ',torch.cuda.get_device_name(0))


# Define mBERT model and tokenizer
device = torch.device("cuda")
model_name = 'bert-base-multilingual-uncased' 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, device_map = 'cuda')
#model = model.to('cuda')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Check pytorch")
print(torch.cuda.is_available())

# Function to get the embedding for a target word 
def get_sentence_embedding(sentence): 
    tokens = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        outputs = model(**tokens)
       # print('outputs', outputs.last_hidden_state.is_cuda)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().to('cuda')  # Assuming you want sentence-level embeddings
   # print('embedding check', embedding.is_cuda)
    return embedding

# My initial attempt to get the embeddings of all sentences before pass them into the model
# Function to get data (sentences), label (corresponiding to sentence, ie, repetitive sequences of 60 language labels), header (60 language labels * 1)
file_path='all_talks_train.tsv'
data = []; labels = []; header = []
def process_dataset(file_path):
    data = []; labels = []; header = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  #read all lines into a list of strings
        header = lines[0].strip().split("\t") #the first line is the header
        lines = lines[1::] #start from the second line, we have sentences
        for line in lines:
            for j, col in enumerate(line.strip().split("\t")):
                data.append(col)
                labels.append(header[j])
    return data,labels,header
data,label,header = process_dataset(file_path)

# Define training parameters
n_steps =100000 #make this number smaller to test if this code works, ideally should have 2000
print_every = 200
validate_every = 200
z_dim = 400
input_dim = 768 #dim of embeddings
batch_size =64 #number of sentence to choose each time {32, 16, 8}
n_clusters = 60   # int(label_mtx.max())
save_dir = '/scratch/jz5213/nmce-models'

# This is the same as in the model where we get the embeddings of teh validation set via mBERT
val_file ='all_talks_dev.tsv'
valdata,vallabels,header = process_dataset(val_file)
def valembedding(valdata,vallabel):
    ids = torch.randint(low=0, high=len(valdata), size=(batch_size,)).to('cuda') #randomly choose 100 sentences
    #datatensor = torch.stack([data[i] for i in ids]) #torch.tensor(data[ids])
    data_select =[valdata[i] for i in ids]#list of sentences (100)
    label_select = [vallabel[i] for i in ids] #list of corresponding labels
    # x should be one example from ted talks
    # check if size is 100 x 1 (100 random sentences)
    emb = []
    for sentence in data_select:
        #embedding = get_sentence_embedding(sentence)
        emb.append(get_sentence_embedding(sentence).to(device))
        #emb.append(embedding.cuda())
    embtensor = torch.stack(emb).to('cuda')
    #print("val",embtensor.device)
    return embtensor, label_select

# Validation set embeddings and their corresponding labels
valembedding,vallabelrandom = valembedding(valdata,vallabels)
#print('vallabelrandom',vallabelrandom)

# Important: in cluster_match/cluster_acc, both data and labels are assumed in tensor type, while our initial labels are just words (eg. en, zh), thus I used their position in the 60 languages sequence to encode the labels (0-59)
lang_to_index = {lang: idx for idx, lang in enumerate(header)}
encoded_vallabelrandom = [lang_to_index[label] for label in vallabelrandom]

# This is a class to generate test_loader which will be used in cluster_match/cluster_acc
class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Get test_loader
dataset = CustomDataset(valembedding,encoded_vallabelrandom )
test_loader = DataLoader(dataset, batch_size=100, shuffle=True)  #load all 100 sentences together
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# Check the data type of valembeddings/vallabels (using position to encode)
for x, y in test_loader:
 #   print(y) #x,y should both be tensors
    x, y = x.float().to(device), y.to(device)
#print("x,y",x.device,y.device)

# My new validate function using cluster_acc (which contains cluster_match/cluster_merge_match)
# I deleted the last part (save-img etc, 3 lines in total) in the cluster_acc function as they are not used to test the performance of the model
def validate():
    net.eval()
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    acc_single, acc_merge=cluster_acc(test_loader,net,device,print_result=True)
   # print('acc_single: ',acc_single,'acc_merge: ', acc_merge)
    return  acc_single,acc_merge

# My initial attempt to test the performance (problem here: the model does not know the true order of the header (labels of 60 languages), thus cannot predict accuralety 
# Trainning starts here
net = SubspaceClusterNetwork(input_dim,z_dim,n_clusters,norm_p=2)

#print("Initial device:", device)  # This should print 'cuda' if CUDA is available and your setup is correct
net = net.to(device)  # Move the model to the GPU if available
#print("Net device:", next(net.parameters()).device)  # This should print 'cuda' if the model has been moved to GPU

optimizer = optim.Adam(net.parameters(),lr=0.01,betas=(0.9,0.99),weight_decay=0.00001)
# after fixing batch size, start playing with lr to choose in {0.01, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005}
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,n_steps)
G_Softmax = Gumble_Softmax(1,straight_through=False)

criterion = TotalCodingRate(eps=0.01) # maybe we should tune this later?
criterion_z = Z_loss()
for i in range(n_steps):

    # take (batch_size) examples from train data
    ids = torch.randint(low=0, high=len(data), size=(batch_size,)).to('cuda') #randomly choose 100 sentences
    #datatensor = torch.stack([data[i] for i in ids]) #torch.tensor(data[ids])
    data_select =[data[i] for i in ids] #list of 100 random selected sentences
    label_select = [label[i] for i in ids] #list of labels coresponding to 100 random seletced sentences (actually don't need it as this is an unsupervised learning model)
    # x should be one example from ted talks
    # check if size is 100 x 1 (100 random sentences)
    emb = []
    for sentence in data_select:
        #embedding = get_sentence_embedding(sentence)
        #emb.append(embedding.cuda())
        emb.append(get_sentence_embedding(sentence).to(device))
    embtensor = torch.stack(emb).to('cuda')
   
    z, logits = net(embtensor) # if it's too big OOM just give datatensor[0:10,:,:]
    loss_z, z_sim = criterion_z(z)
    z_sim = z_sim.mean()
    prob = G_Softmax(logits)
    z, prob = chunk_avg(z,n_chunks=2,normalize=True), chunk_avg(prob,n_chunks=2)
    # why n_chunks=2? 
    loss = criterion(z)
    loss_list = [loss.item(),loss.item()]
    
    loss += 20*loss_z # should this be 20?
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i%print_every == 0:   
        print('{} steps done, loss c {}, loss d {}, z sim {}'.format(i+1,loss_list[0],loss_list[1],z_sim.item()))
    if i%validate_every == 0:
        validate()
       # print('{} steps done,'.format(i+1),'acc_single: ',acc_single,'acc_merge: ', acc_merge)
        #    acc, preds = validate()

        net.train()
    #print("N device:", next(net.parameters()).device) 

#this is probably where I can store my model and restart again if this model is too big
#utils.save_ckpt(save_dir, net, optimizer, scheduler, step)
print("training complete.")

