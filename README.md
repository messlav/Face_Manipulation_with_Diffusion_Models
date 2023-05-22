# Face Manipulation with Diffusion Models
****
This code is based on [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP) and [Efficient DiffusionCLIP](https://github.com/quickjkee/eff-diff-edit)
****
# Run code
****
### 1. Preparation

* _Install required dependencies_
```commandline
# Clone the repo
git clone https://github.com/messlav/Face_Manipulation_with_Diffusion_Models

# Install dependencies
pip install ftfy regex tqdm
pip install lmdb
pip install pynvml
pip install git+https://github.com/openai/CLIP.git

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
```
* _Download pretrained diffusion models_

  * Pretrained diffusion models on CelebA-HQ-256, LSUN-Church-256 are automatically downloaded in the code.

  * For AFHQ-Dog-256 and ImageNet-512, please download the corresponding models ([ImageNet](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt), [AFHQ-Dog](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&cid=72419B431C262344&id=72419B431C262344%21103832&parId=72419B431C262344%21103807&o=OneUp)) and put them into the ```./pretrained``` folder


* _Download datasets_ (this part can be skipped if you have your own training set, please see the second section for details)
   * For CelebA-HQ and AFHQ-Dog you can use the following code:    
  ``` commandline
  # CelebA-HQ 256x256
  bash data_download.sh celeba_hq .
  
  # AFHQ-Dog 256x256
  bash data_download.sh afhq .
  ```
  * For [LSUN-Church](https://www.yf.io/p/lsun) and [ImageNet](https://image-net.org/index.php), you can download them from the original sources and put them into `./data/lsun` or `./data/imagenet`.

### 2. Running
1. Select the config for the particular dataset: ```celeba.yaml / afhq.yaml / church.yaml / imagenet.yaml```.
2. Select the desired manipulation from the list. The list of available textual transforms for each dataset is [here](/utils/text_dic.py).\
Note that you can also add your own transforms to this file.

Below we provide the commands for different settings:

* _Prelearned image manipulations (**dataset training** and **dataset test**)_ \
This command adapts the pretrained model using images from the training set and applies the learned transform to the test images. 
The following command uses 50 CelebA-HQ images for training and evaluation:

```commandline
python3 main.py --clip_finetune         \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr makeup \
               --bs_test 1            \
               --n_train_img 50       \
               --n_precomp_img 50 \
               --single_image 0 \
               --n_test_img 10        \
               --n_iter 1         \
               --t_0 525             \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 6        \
               --lr_clip_finetune 1e-5  \
               --id_loss_w 0 \
               --clip_loss_w 3 \
               --l1_loss_w 0.9 \
               --scheduler 0 \
               --clip_model_name ViT-B/32 \
               --wandb 1 \
               --wandb_run_name makeup_freeze_attn_temb_all_sparse1_lora_32_128 \
               --wandb_test_my_image True \
               --name_of_freeze_function freeze_attn_temb_all_sparse1 \
               --lora 1 \
               --lora_rank 32 \
               --lora_alpha 128 \
```
