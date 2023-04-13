"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from model import *
from dataloader import data_generator
from trainer import Trainer
from rtpt import RTPT
from torchinfo import summary

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument(
    "--run_description", default="run1", type=str, help="Experiment Description"
)
parser.add_argument("--seed", default=42, type=int, help="seed value")

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument(
    "--training_mode", default="pre_train", type=str, help="pre_train, fine_tune_test"
)

parser.add_argument(
    "--pretrain_dataset",
    default="SleepEEG",
    type=str,
    help="Dataset of choice: SleepEEG, FD_A, HAR, ECG",
)
parser.add_argument(
    "--target_dataset",
    default="Epilepsy",
    type=str,
    help="Dataset of choice: Epilepsy, FD_B, Gesture, EMG",
)

parser.add_argument(
    "--logs_save_dir", default="../exp_logs", type=str, help="saving directory"
)
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument("--device_id", default="0", type=int, help="cuda device id")
parser.add_argument(
    "--home_path", default=home_dir, type=str, help="Project home directory"
)

parser.add_argument(
    "--model",
    default="1D-ResNet-from-paper",
    type=str,
    choices=["1D-ResNet-from-paper", "New-Transformer"],
    help="The model to use as a backbone",
)

args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device(f"cuda:{args.device_id}")
    print(device)
else:
    device = torch.device("cpu")
print("We are using %s now." % device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + "_2_" + str(targetdata)


method = "TF-C"
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
import_config = f'from config_files.{pretrain_dataset.replace("-","_")}_Configs import Config as Configs'
exec(import_config)
print(f"importing config: {import_config}")
configs = Configs()

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(
    logs_save_dir,
    experiment_description,
    run_description,
    training_mode + f"_seed_{SEED}_{args.model}",
)
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(
    experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
)
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f"Pre-training Dataset: {pretrain_dataset}")
logger.debug(f"Target (fine-tuning) Dataset: {targetdata}")
logger.debug(f"Method:  {method}")
logger.debug(f"Mode:    {training_mode}")
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"../../datasets/{pretrain_dataset}"
targetdata_path = f"../../datasets/{targetdata}"
subset = False  # if subset= true, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(
    sourcedata_path, targetdata_path, configs, training_mode, subset=subset
)
logger.debug("Data loaded ...")


rtpt = RTPT(
    name_initials="MK",
    experiment_name=f'TFDC-{args.training_mode}-{args.pretrain_dataset}{f"2{args.target_dataset}"if args.training_mode == "fine_tune_test" else ""}',
    max_iterations=configs.num_epoch,
)
# Load Model
# Start the RTPT tracking
rtpt.start()

"""Here are two models, one basemodel, another is temporal contrastive model"""
if args.model == "1D-ResNet-from-paper":
    TFC_model = TFC_Original(configs)
elif args.model == "New-Transformer":
    TFC_model = TFC(configs).to(device)
else:
    raise NotImplementedError(f"Model {args.model} is not implemented.")
# Log information on the model
print()  # newline
print(f"The model being used:\n{TFC_model}")
print()  # newline
in_data = next(iter(train_dl))  # See trainer for an interpretation of the list
summary(TFC_model, input_data=[in_data[0], in_data[3]], device=device)
print()  # newline
# Configure:
TFC_model = TFC_model.to(device)  # torch.compile() makes it slower
classifier = target_classifier(configs).to(device)

if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(
        os.path.join(
            logs_save_dir,
            experiment_description,
            run_description,
            f"pre_train_seed_{SEED}_{args.model}",
            "saved_models",
        )
    )
    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(
    TFC_model.parameters(),
    lr=configs.lr,
    betas=(configs.beta1, configs.beta2),
    weight_decay=5e-4,
)
classifier_optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=configs.lr,
    betas=(configs.beta1, configs.beta2),
    weight_decay=5e-4,
)

# Trainer
Trainer(
    TFC_model,
    model_optimizer,
    classifier,
    classifier_optimizer,
    train_dl,
    valid_dl,
    test_dl,
    device,
    logger,
    configs,
    experiment_log_dir,
    training_mode,
    rtpt,
)

logger.debug(f"Training time is : {datetime.now()-start_time}")
