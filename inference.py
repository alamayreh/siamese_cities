import torch
import logging
import torchvision
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from train_sigmoid import SiameseNetwork, SiameseNetworkDataset
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm

#export CUDA_VISIBLE_DEVICES=4,5,6,7

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        default=Path("models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/ckpts/epoch_621.ckpt"),
        #default=Path("models/resnet101_128_sigmoid_acc_noFlatren/220915-0730/ckpts/epoch_15.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        default=Path("models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--checkpoint_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/ckpts/epoch_15.ckpt"),
        #default=Path("models/resnet101_64_embedding/220912-0923/ckpts/epoch_34.ckpt"),
        #default=Path("models/resnet101_64_embedding_Adam/220919-0156/ckpts/epoch_100.ckpt"),
        default=Path("models/resnet101_32_embedding_SGD/220919-0838/ckpts/epoch_1858.ckpt"),
        #default=Path('models/resnet101_1024_embedding_margin_1_SGD/220920-1046/ckpts/epoch_346.ckpt'),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/tb_logs/version_0/hparams.yaml"),
        #default=Path("models/resnet101_64_embedding/220912-0923/tb_logs/version_0/hparams.yaml"),
        default=Path("models/resnet101_32_embedding_SGD/220919-0838/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("dataset/cities/test"),
        help="Folder containing test set images.",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


def test_dataloader(image_dir, batch_size, num_workers):

    # logging.info("val_dataloader")

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir)

    dataset = SiameseNetworkDataset(
        imageFolderDataset=DatasetFolder_test, transform=tfm_test)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading model from : {args.checkpoint_sigmoid}")

    model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_sigmoid),
        hparams_file=str(args.hparams_sigmoid),
        map_location=None,
    )

    if args.gpu and torch.cuda.is_available():
        model_sigmoid.cuda()

    #model_sigmoid.eval()
    logging.info(f"Loading test data : {args.image_dir}")

    test_dataloader = test_dataloader(args.image_dir, args.batch_size, args.num_workers)

    dataset_length = len(test_dataloader.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(test_dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    correct = 0 

    for im1,im2,target in tqdm(test_dataloader):
        if args.gpu:
            im1 = im1.cuda()
            im2 = im2.cuda()
            target = target.cuda()    

        #output_model = model(Variable(x0).cuda(), Variable(x1).cuda())
        output_model = model_sigmoid(im1, im2)
        
        #print(output_model)
        
        pred = torch.where(output_model > 0.5, 1, 0)

        #print('pred', pred)
        #print('target', target)
        correct += pred.eq(target.view_as(pred)).sum().item()


    val_acc_sigmoid =100. * correct / dataset_length



    logging.info(f"Loading model from : {args.checkpoint_contrastive}")

    model_contrastive = SiameseNetwork_contrastive.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_contrastive),
        hparams_file=str(args.hparams_contrastive),
        map_location=None,
    )

    if args.gpu and torch.cuda.is_available():
        model_contrastive.cuda()
    
    #model_contrastive.eval()

    correct = 0 
    correct_cosine = 0

    cos = torch.nn.CosineSimilarity()

    for im1,im2,target in tqdm(test_dataloader):
        if args.gpu:
            im1 = im1.cuda()
            im2 = im2.cuda()
            target = target.cuda()    

        output1, output2 = model_contrastive(im1, im2)
       
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        cosine_distance =  torch.tensor([1]).cuda() - cos(output1, output2)

        pred_cosine =  torch.where(cosine_distance > 0.5, 1, 0)

        pred = torch.where(euclidean_distance > 0.5, 1, 0)

        correct += pred.eq(target.view_as(pred)).sum().item()

        correct_cosine += pred_cosine.eq(target.view_as(pred_cosine)).sum().item()


    val_acc_contrastive =100. * correct / dataset_length

    val_acc_cosine = 100. * correct_cosine / dataset_length




    print(f"val_acc sigmoid: {val_acc_sigmoid}")

    print(f"val_acc contrastive: {val_acc_contrastive}")

    print(f"val_acc cosine: {val_acc_cosine}")