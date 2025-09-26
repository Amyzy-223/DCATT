import time
import argparse
import torch



def set_config():
    ## set hyperparameters
    t = time.localtime()
    foldname = str(t.tm_yday).zfill(3) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + str(t.tm_sec).zfill(3) + '/'

    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--lr', default=1E-4, type=float, help='learning rate 1E-1')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1E-4, type=float, help='weight decay')
    parser.add_argument('--epoch_num', default=300, type=int, help='num epochs')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler: only cosine')
    parser.add_argument('--seed', default=3407, type=int, help='random seed 3407 42')
    parser.add_argument('--esPatience', default=15, type=int, help='early stopping patience')
    parser.add_argument('--dropout', default=0.5, type=float, help='drop out')
    parser.add_argument('--criterion', default='entropy', type=str, help='loss function: entropy')
    parser.add_argument('--alpha', default=3, type=float, help='ratio of major sample to minor sample')

    parser.add_argument('--dataset', default='SEED_DCATT', type=str, help='dataset name')
    parser.add_argument('--SubTestFlag', default='LOSO', type=str, help='subtest flag: DEP/LOSO (dependent/independent)')
    parser.add_argument('--subject', default=0, type=int, help='subject number')

    parser.add_argument('--modelname', default='DCATT', type=str, help='model name')
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden_dim")
    parser.add_argument("--end_dim", type=int, default=1024, help="end_dim")
    parser.add_argument("--layer", type=int, default=2, help="DCATT STblock layer")



    # do not change
    parser.add_argument("--root_dir", type=str, default="./train_result_test/", help="root dir")
    parser.add_argument('--kfold', default=5, type=int, help='fold number')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), type=str, help='device')
    parser.add_argument("--result_dir", type=str, default="./result/", help="result dir")
    parser.add_argument("--model_dir", type=str, default="./model/", help="model dir")
    parser.add_argument("--plot_dir", type=str, default="./plot/", help="plot dir")
    # parser.add_argument("--data_dir", type=str, default=r"F:\AMY\研究生\组里相关\04[横向项目]\1ZGH\12时序e2e\LoadData\datas", help="Data directory.")
    parser.add_argument("--data_dir", type=str,
                        default=r"./datas/loaddata",
                        help="Data directory.")
    parser.add_argument("--split_ratio_train", type=float, default=0.8, help="split ratio")

    args = parser.parse_args()
    args.criterion = (args.criterion, args.alpha)
    print('criterion: ', args.criterion)


    return args, foldname