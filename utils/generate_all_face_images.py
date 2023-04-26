from diffusionclip import DiffusionCLIP
from main import dict2namespace
import re
import argparse
import os
import yaml
from PIL import Image
import warnings

warnings.filterwarnings(action='ignore')

device = 'cuda'


def gen_all_face_images(model_name, t_0, n_iter):
    model_path = os.path.join('checkpoint', f"{model_name}-{n_iter - 1}.pth")
    if n_iter == -1:
        model_path = os.path.join('checkpoint', f"{model_name}.pth")

    align_face = True
    img_path = 'imgs/passport_photo.png'
    exp_dir = f"runs/MANI_{img_path.split('/')[-1]}_align{align_face}"
    os.makedirs(exp_dir, exist_ok=True)

    degree_of_change = 1
    # Test arg, config
    n_inv_step = 40
    n_test_step = 6
    n_iter_forward = 1
    args_dic = {
        'lr_clip_finetune': 1e-5,
        'sch_gamma': None,
        'clip_model_name': 'ViT-B/16',
        'config': 'celeba.yml',
        'n_train_step': 6,
        't_0': t_0,
        'n_inv_step': int(n_inv_step),
        'n_test_step': int(n_test_step),
        'sample_type': 'ddim',
        'eta': 0.0,
        'bs_test': 1,
        'model_path': model_path,
        'img_path': img_path,
        'deterministic_inv': 1,
        'hybrid_noise': 0,
        'n_iter': n_iter_forward,
        'align_face': align_face,
        'image_folder': exp_dir,
        'model_ratio': degree_of_change,
        'edit_attr': None, 'src_txts': None, 'trg_txts': None,
        'name_of_freeze_function': None,
    }
    args = dict2namespace(args_dic)

    with open(os.path.join('configs', args.config), 'r') as f:
        config_dic = yaml.safe_load(f)
    config = dict2namespace(config_dic)
    config.device = device

    # Edit
    runner = DiffusionCLIP(args, config)
    n_result = 1
    img_orig = Image.open(os.path.join(exp_dir, '0_orig.png'))
    img_orig = img_orig.resize((int(img_orig.width), int(img_orig.height)))
    runner.edit_one_image()
    for j in range(n_iter - 2, -2, -1):
        # Result
        # print()
        grid = Image.new("RGB", (img_orig.width * (n_result + 1), img_orig.height))
        grid.paste(img_orig, (0, 0))
        for i in range(n_result):
            img = Image.open(os.path.join(exp_dir,
                                          f"3_gen_t{t_0}_it0_ninv{n_inv_step}_ngen{n_test_step}_mrat{degree_of_change}_{model_path.split('/')[-1].replace('.pth', '')}.png"))
            img = img.resize((int(img.width), int(img.height)))
            grid.paste(img, (int(img.height * (i + 1)), 0))

        grid.show()
        print('ITER =', j + 1)

        if j != -1:
            model_path = os.path.join('checkpoint', f"{model_name}-{j}.pth")
            runner.set_model(model_path)
            runner.edit_one_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='name of the model in the checkpoint folder')
    # parser.add_argument('--t_0', type=int, required=True, help='t_0 of model')
    parser.add_argument('--n_iter', type=int, required=False, default=-1,
                        help='num of models created with specified model name')
    args = parser.parse_args()
    model_name = args.model_name
    n_iter = args.n_iter

    pattern = r'_t(\d+)_'
    match = re.search(pattern, model_name)
    if match:
        t_0 = int(match.group(1))
    else:
        print('t_0 should be specified in model name')
        exit(1)

    gen_all_face_images(model_name, t_0, n_iter)
