****
## Dataset
Firstly, download Celeba-HQ dataset
```
bash data_download.sh celeba_hq .
```
****

****
## Images manipulation
As an example, you can run the following
```
python main.py --clip_finetune         \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr angry \
	       --bs_test 1            \
               --n_train_img 50       \
               --n_precomp_img 50 \
               --single_image 0 \
               --n_test_img 35        \
               --n_iter 5         \
               --t_0 400             \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 6        \
               --lr_clip_finetune 3e-6  \
               --id_loss_w 0.0       \
               --clip_loss_w 3 \
               --l1_loss_w 0.8 \    
               --scheduler 0 

```
You can run different transformations, check the full list in the utils/text_dic.py
****

****
## Single image editing
```
python main.py --clip_finetune         \
               --number_of_image 0 \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr neanderthal \
               --n_train_img 1       \
               --n_precomp_img 1 \
               --single_image 1 \
               --n_test_img 1        \
               --n_iter 20         \
               --t_0 400             \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 6        \
               --lr_clip_finetune 2e-6  \
               --id_loss_w 0.0       \
               --clip_loss_w 3 \
               --l1_loss_w 0.1 \
               --scheduler 0 
```
****
