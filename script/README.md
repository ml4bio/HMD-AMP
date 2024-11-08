# HMD-AMP training procedure
Here contains the training scripts for HMD-AMP.

## AMP/non-AMP prediction

amp_extract_repr.py is for fine-tuning the protein language model and extract embeddings from the fine-tuned model. If the fine-tuned model is available in advance, it will directly generate the embeddings.
```
usage: python amp_extract_repr.py --training_data FASTA --training_label label.npy --ftmodel_save_path MODEL_DIRECTORY --emb_path EMBEDDINGS_SAVE_DIRECTORY [--lr learning_rate] [--epochs EPOCHS] [--batch_size BATCH_SIZE]

optional arguments:
--training_data
                Path of the training data fasta file
--training_label
                Path of the training labels, in .npy format
--ftmodel_save_path
                Fine-tuned model saving directory
--emb_path
                Directory to save the embeddings and labels
--lr
                Initial learning rate
--epochs
                Upper training epoch limit
--batch_size
                Batch size
```


amp_train.py is for training the AMP/non-AMP classifier with the extracted embeddings as input.

```
usage: python amp_train.py --training_emb EMBEDDING_DIRECTORY --clsmodel_save_path MODEL_SAVE_PATH

optional arguments:
--training_emb
                Directory of the training embeddings and labels
--clsmodel_save_path
                The trained prediction model saving directory, you should not create it in advance
```

## AMP target groups prediction

target_extract_repr.py is for fine-tuning the protein language model and extract embeddings from the fine-tuned model. 5-fold cross validation strategy is applied for training and evaluating, so it will train 5 models for each target group. If you do not want to train 5 times, please modify the for loop under `if __name__ == '__main__':`. If the fine-tuned model is available in advance, it will directly generate the embeddings.

```
usage: python target_extract_repr.py --target TARGET --training_data FASTA --training_label label.npy --ftmodel_save_path MODEL_DIRECTORY --emb_path EMBEDDINGS_SAVE_DIRECTORY [--lr learning_rate] [--epochs EPOCHS] [--batch_size BATCH_SIZE]

optional arguments:
--target         
                Specify the target group to predict: Gram+, Gram-, Mammalian_Cell, Virus, Fungus, or Cancer
--data_path
                Path of the training data fasta file
--label_path
                Path of the training labels, in .npy format
--ftmodel_save_path
                Fine-tuned model saving directory
--emb_path
                Directory to save the embeddings and labels
--lr
                Initial learning rate
--epochs
                Upper training epoch limit
--batch_size
                Batch size
```


target_train.py is for training the specific AMP target group predictor with the extracted embeddings as input.

```
usage: python target_train.py --target TARGET --training_emb EMBEDDING_DIRECTORY --clsmodel_save_path MODEL_SAVE_PATH

optional arguments:
--target         
                Specify the target group to predict: Gram+, Gram-, Mammalian_Cell, Virus, Fungus, or Cancer
--training_emb
                Directory of the training embeddings and labels
--clsmodel_save_path
                The trained prediction model saving directory, you should not create it in advance
```