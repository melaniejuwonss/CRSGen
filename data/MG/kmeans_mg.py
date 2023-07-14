import torch
import numpy as np
from kmeans_pytorch import kmeans
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader


# data
# data_size, dims, num_clusters = 20, 2, 3
# x = np.random.randn(data_size, dims)
# x = torch.from_numpy(x)  # [Num, dim]
# threshold = 2
# originalIdx = torch.range(0, data_size - 1, dtype=int)
# target_id = [""] * data_size


# kmeans
def runKmeans(x, num_clusters, orgIdx):
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )

    # print(cluster_ids_x)
    # print(cluster_centers)

    return cluster_ids_x, orgIdx


# recursive kmeans
class RecursiveKmeans:
    def __init__(self, input, num_clusters):
        self.num_clusters = num_clusters
        self.input = input
        self.data_size = self.input.size(0)
        self.originalIdx = torch.range(0, self.data_size - 1, dtype=int)
        self.target_id = [""] * self.data_size

    def process(self, begin, cluster_ids=None, orgIdx=None, startIdx=0):
        if begin:
            cluster_ids, orgIdx = runKmeans(self.input, self.num_clusters, self.originalIdx)
            begin = False

        if startIdx == self.num_clusters:
            return

        cluster_index = (cluster_ids == startIdx).nonzero(as_tuple=True)[0]
        num_index = cluster_index.size()  #####
        org_tmp = orgIdx
        cluster_tmp = cluster_ids
        orgIdx = orgIdx[cluster_index]
        for j in range(num_index[0]):
            self.target_id[orgIdx[j]] += str(startIdx)
        if num_index[0] >= self.num_clusters:
            cluster_ids, orgIdx = runKmeans(self.input[orgIdx], self.num_clusters, orgIdx)
            self.process(False, cluster_ids, orgIdx, 0)
            self.process(False, cluster_tmp, org_tmp, startIdx + 1)
            # saveIdx(cluster_ids)
        else:
            self.process(False, cluster_ids, org_tmp, startIdx + 1)
            return


class reviewInformation(Dataset):
    def __init__(self, tokenizer, bert_config, num_reviews):
        self.content_data = json.load(open('mgcrs_allknowledges.json', 'r', encoding='utf-8'))
        self.id2knowledge = {v: k for k, v in self.content_data.items()}
        # self.movie2name = json.load(open('movie2name.json', 'r', encoding='utf-8'))
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.data_samples = dict()
        self.num_reviews = num_reviews
        self.max_review_len = 128
        self.read_data()
        self.key_list = list(self.data_samples.keys())

    def read_data(self):
        for content in self.id2knowledge:
            idx = content
            knowledge = self.id2knowledge[idx]

            tokenized_knowledge = self.tokenizer(knowledge,
                                                 max_length=self.max_review_len,
                                                 padding='max_length',
                                                 truncation=True,
                                                 add_special_tokens=True)

            self.data_samples[idx] = {
                "knowledge": tokenized_knowledge.input_ids,
                "knowlege_mask": tokenized_knowledge.attention_mask
            }

    def __getitem__(self, item):
        idx = self.key_list[item]
        knowledge_token = self.data_samples[idx]['knowledge']
        knowledge_mask = self.data_samples[idx]['knowlege_mask']

        idx = torch.tensor(int(idx)).to(0)
        knowledge_token = torch.LongTensor(knowledge_token).to(0)  # [R, L]
        knowledge_mask = torch.LongTensor(knowledge_mask).to(0)  # [R, L]

        return idx, knowledge_token, knowledge_mask

    def __len__(self):
        return len(self.data_samples)


class ReviewEmbedding(nn.Module):
    def __init__(self, num_reviews, max_review_len, token_emb_dim, bert_model):
        super(ReviewEmbedding, self).__init__()
        self.bert_model = bert_model
        self.num_reviews = num_reviews
        self.max_review_len = max_review_len
        self.token_emb_dim = token_emb_dim

    def forward(self, knowledge_token, knowledge_mask):
        knowledge_rep = self.bert_model(input_ids=knowledge_token,
                                        attention_mask=knowledge_mask).last_hidden_state[:, 0, :]  # [M, d]

        return knowledge_rep.tolist()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_config = AutoConfig.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(0)

    dataset = reviewInformation(tokenizer, bert_config, 0)
    print("===============Dataset Done===============")
    review_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = ReviewEmbedding(1, 128, bert_config.hidden_size, bert_model).to(0)

    knowledg_embedding, knowledge_id = [], []

    for idx, knowledge_token, knowledge_mask in tqdm(review_dataloader,
                                                     bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        knowledg_embedding.extend(model.forward(knowledge_token, knowledge_mask))
        knowledge_id.extend(idx.tolist())
    print("===============Review Embedding Done===============")
    # Recursive clustering
    recur = RecursiveKmeans(torch.tensor(knowledg_embedding), 10)
    recur.process(True)
    print("===============Recursive KMeans Done===============")
    # Create final target ids
    idDict = dict()
    for i in range(len(recur.target_id)):
        if recur.target_id[i] not in idDict.keys():
            idDict[recur.target_id[i]] = 0
            recur.target_id[i] += str(idDict[recur.target_id[i]])

        else:
            idDict[recur.target_id[i]] += 1
            recur.target_id[i] += str(idDict[recur.target_id[i]])

    final_target_id = recur.target_id

    # Save {crs_id : target_id}
    saveDict = dict()
    for i in range(len(knowledge_id)):
        saveDict[knowledge_id[i]] = final_target_id[i]
    with open('knowledge_kmeansid.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(saveDict, indent=4))
