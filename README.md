# CRSGen

## 1) Dataset
### crsid2id.json
* crs id 를 target item id 로 mapping
* {crs_id : target_item_id}

### review_{#review}.json
* indexing stage training 으로 사용
* {"context_tokens" : "review text", "item" : "target_item_id"}

### {train, test, valid}.json
* recommendation stage training, test, valid 으로 사용
* {"context_tokens" : [Dialog Lists], "item" : "target_item_id"}

### train_review_{#review}.json
* multi-task recommendation stage training 으로 사용
* train.json | review_{#review}.json 을 합쳐놓은 결과
* 어떤 dataset 인지 구분하기 위해, review 의 경우 맨 앞에 "Review: " prepend

## 2) Argument
* name: log name
* model_name: t5-large
* max_dialog_len: 최대 dialog / review 개수 조절
* num_index_epochs: indexing stage 수행 시 epochs 수 조절
* num_train_epochs: recommendation stage epochs 수 주절
* train_batch_size: training batch size
* eval_batch_size: test batch_size
* learning_rate: learning rate
* num_reviews: review 개수 조절
* prefix: "Dialog: ", "Review: " prepend 할지 여부 결정
* train_type
  * 0: multi-task 수행 (train_review_{#review}.json)
  * 1: indexing -> multi-task 수행 (review_{#review}.json --> train_review_{#review}.json)
  * 2: indexing -> training 수행 (review_{#review.json} --> train.json)
