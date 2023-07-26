import json


def create_review_with_index(num):
    content_data = json.load(open('../content_data.json', 'r', encoding='utf-8'))[0]
    crs2kmeansid = json.load(open('crsid2id.json', 'r', encoding='utf-8'))

    saveList = list()
    for data in content_data:
        crs_id = data['crs_id']
        reviews = data['review']
        for idx, review in enumerate(reviews):
            if idx >= num:
                break
            else:
                kmeans_id = crs2kmeansid[crs_id]
                saveList.append({"context_tokens": review, "item": kmeans_id})
    with open(f'review_{num}.json', 'w', encoding='utf-8') as wf:
        wf.write(json.dumps(saveList, indent=4))


def create_train_valid_test():
    datasets = ['../../originalRedial/train.json', '../../originalRedial/valid.json', '../../originalRedial/test.json']
    mode = ['train', 'valid', 'test']
    crs2kmeansid = json.load(open('crsid2id.json', 'r', encoding='utf-8'))
    for dataidx, data in enumerate(datasets):
        saveList = list()
        dialogs = json.load(open(data, 'r', encoding='utf-8'))
        for dialog in dialogs:
            context_token_lists = dialog['context_tokens']
            new_context_token = ""
            for context_token in context_token_lists:
                token_list = context_token.split(' ')
                for idx, token in enumerate(token_list):
                    if token == "<movie>" and token_list[idx + 1].isdigit():
                        kmeans_id = crs2kmeansid[token_list[idx + 1]]
                        token_list[idx + 1] = kmeans_id
                    new_context_token += token + " "
                new_context_token += "</s> "
            item = dialog['item']
            kmeans_item = crs2kmeansid[item]

            saveList.append({'context_tokens': [new_context_token], 'item': kmeans_item})
        with open(mode[dataidx] + '.json', 'w', encoding='utf-8') as wf:
            wf.write(json.dumps(saveList, indent=4))


def mergeTrainandIndex():
    for i in range(7, 8):
        result = list()
        filenames = ['train.json', f'review_{i}.json']
        for f1 in filenames:
            with open(f1, 'r') as infile:
                tmp_file = json.load(infile)
                if f1 == f"review_{i}.json":
                    for detail in tmp_file:
                        newDict = dict()
                        newDict['context_tokens'] = ["Review: " + detail['context_tokens']]
                        newDict['item'] = detail['item']
                        result.append(newDict)
                else:
                    # for detail in tmp_file:
                    #     newDict = dict()
                    #     newContext = "</s>".join(detail['context_tokens'])
                    #     newDict['context_tokens'] = "Dialog: " + newContext
                    #     newDict['item'] = detail['item']
                    result.extend(tmp_file)

        with open(f'train_review_{i}.json', 'w') as outfile:
            outfile.write(json.dumps(result, indent=4))


def create_all_item_id():
    crs2kmeansid = json.load(open('crsid2id.json', 'r', encoding='utf-8'))
    all_item_list = []
    for key, value in crs2kmeansid.items():
        all_item_list.append(value)

    with open(f'item.json', 'w') as outfile:
        outfile.write(json.dumps(all_item_list, indent=4))


for i in range(7, 8):
    create_review_with_index(i)  # Create {review : idx}
create_train_valid_test()  # Create train, test
mergeTrainandIndex()  # Merge train and index
create_all_item_id()
