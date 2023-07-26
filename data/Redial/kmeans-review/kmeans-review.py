import torch
import numpy as np
from kmeans_pytorch import kmeans
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys


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
        X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0')
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
    def __init__(self, tokenizer, bert_config, num_reviews, max_review_len):
        self.content_data = json.load(open('content_data.json', 'r', encoding='utf-8'))[0]
        self.movie2name = json.load(open('movie2name.json', 'r', encoding='utf-8'))
        self.tokenizer = tokenizer
        self.bert_config = bert_config
        self.data_samples = dict()
        self.num_reviews = num_reviews
        self.max_review_len = max_review_len
        self.read_data()
        self.key_list = list(self.data_samples.keys())

    def read_data(self):
        for content in self.content_data:
            review_list, review_mask_list = [], []
            crs_id = content['crs_id']
            title = self.movie2name[crs_id][1]
            meta = self.movie2name[crs_id][2]
            reviews = content['review'][:self.num_reviews]

            seed_keyword = "title:" + title + " " + meta
            if len(reviews) != 0:
                sampled_reviews = [seed_keyword + " " + review for review in reviews]
                tokenized_reviews = self.tokenizer(sampled_reviews, max_length=self.max_review_len,
                                                   padding='max_length',
                                                   truncation=True,
                                                   add_special_tokens=True)
                tokenized_title = self.tokenizer(seed_keyword,
                                                 max_length=self.max_review_len,
                                                 padding='max_length',
                                                 truncation=True,
                                                 add_special_tokens=True)
            else:
                sampled_reviews = []
                tokenized_title = self.tokenizer(seed_keyword,
                                                 max_length=self.max_review_len,
                                                 padding='max_length',
                                                 truncation=True,
                                                 add_special_tokens=True)
            for i in range(min(len(sampled_reviews), 1)):
                review_list.append(tokenized_reviews.input_ids[i])
                review_mask_list.append(tokenized_reviews.attention_mask[i])
            for i in range(1 - len(sampled_reviews)):
                # zero_vector = [0] * self.max_review_len
                review_list.append(tokenized_title.input_ids)
                review_mask_list.append(tokenized_title.attention_mask)

            self.data_samples[crs_id] = {
                "review": review_list,
                "review_mask": review_mask_list,
                "title": tokenized_title.input_ids,
                "title_mask": tokenized_title.attention_mask
            }

    def __getitem__(self, item):
        idx = self.key_list[item]
        title = self.data_samples[idx]['title']
        title_mask = self.data_samples[idx]['title_mask']
        review_token = self.data_samples[idx]['review']
        review_mask = self.data_samples[idx]['review_mask']

        idx = torch.tensor(int(idx)).to(0)
        title = torch.LongTensor(title).to(0)  # [L, ]
        title_mask = torch.LongTensor(title_mask).to(0)  # [L, ]
        review_token = torch.LongTensor(review_token).to(0)  # [R, L]
        review_mask = torch.LongTensor(review_mask).to(0)  # [R, L]
        # num_review_mask = torch.tensor([1] * num_reviews + [0] * (self.args.n_review - num_reviews)).to(
        #     self.args.device_id)

        return idx, title, title_mask, review_token, review_mask  # , num_review_mask

    def __len__(self):
        return len(self.data_samples)


class ReviewEmbedding(nn.Module):
    def __init__(self, num_reviews, max_review_len, token_emb_dim, bert_model):
        super(ReviewEmbedding, self).__init__()
        self.bert_model = bert_model
        self.num_reviews = num_reviews
        self.max_review_len = max_review_len
        self.token_emb_dim = token_emb_dim

    def forward(self, title, title_mask, review, review_mask):
        if self.num_reviews != 0:
            review = review.view(-1, self.max_review_len)  # [B X R, L]
            review_mask = review_mask.view(-1, self.max_review_len)  # [B X R, L]
            review_emb = self.bert_model(input_ids=review, attention_mask=review_mask).last_hidden_state[:, 0,
                         :].view(-1, self.num_reviews, self.token_emb_dim)  # [M X R, L, d]  --> [M, R, d]
            title_emb = self.bert_model(input_ids=title,
                                        attention_mask=title_mask).last_hidden_state[:, 0, :]  # [M, d]
            # query_embedding = title_emb
            # item_representations = self.item_attention(review_emb, query_embedding, num_review_mask)
            review_rep = (torch.mean(review_emb, dim=1) + title_emb)
        elif self.num_reviews == 0:
            title_emb = self.bert_model(input_ids=title,
                                        attention_mask=title_mask).last_hidden_state[:, 0, :]  # [M, d]
            review_rep = title_emb

        return review_rep.tolist()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_config = AutoConfig.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(0)
    max_review_len = 512
    batch_size = 32
    num_review = 5

    sys.setrecursionlimit(10 ** 6)

    dataset = reviewInformation(tokenizer, bert_config, num_review, max_review_len)
    print("===============Dataset Done===============")
    review_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = ReviewEmbedding(1, max_review_len, bert_config.hidden_size, bert_model).to(0)

    review_embedding, movie_crs_id = [], []

    for idx, title, title_mask, review, review_mask in tqdm(review_dataloader,
                                                            bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        review_embedding.extend(model.forward(title, title_mask, review, review_mask))
        movie_crs_id.extend(idx.tolist())
    print("===============Review Embedding Done===============")
    # Recursive clustering
    recur = RecursiveKmeans(torch.tensor(review_embedding), 10)
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
    for i in range(len(movie_crs_id)):
        saveDict[movie_crs_id[i]] = final_target_id[i]
    with open('yesconcat5/crsid2id.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(saveDict, indent=4))
