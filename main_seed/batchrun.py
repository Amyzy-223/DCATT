import os
import json
import time
import subprocess


def command_run(args, sys_time):
    print("===============================")
    script = 'main.py'
    command = 'python ' + script + " ".join(args)
    print(command)
    exit_code = os.system(command)
    if exit_code:
        error_dict = {'time': sys_time, 'command': command}
        write_dict(error_dict)



def get_sys_time():
    t = time.localtime()
    sys_time = str(t.tm_yday).zfill(3) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + str(t.tm_sec).zfill(3)
    return sys_time

# print('\nEEGNet')  # dataset SEED_e2e lr 1e-4
# print('\nDGCNN')  # dataset SEED_DGCNN lr 1e-3
# print('\nTSCeption')  # dataset SEED_e2e lr 1e-3
# print('\nEmT')  # dataset SEED_EmT lr 3e-4

def model_init(model):
    if model == 'DGCNN':
        lr = 1E-3
        dataset = 'SEED_DGCNN'
    elif model == 'TSCeption':
        lr = 1E-3
        dataset = 'SEED_e2e'
    elif model == 'EmT':
        lr = 3E-4
        dataset = 'SEED_EmT_final'
    elif model == 'EEGNet':
        lr = 1E-4
        dataset = 'SEED_e2e'
    elif model == 'DCATT':
        lr = 1E-4
        dataset = 'SEED_DCATT'
    elif model == 'Conformer':
        lr = 2E-4
        dataset = 'SEED_Conformer'
    return lr, dataset

def write_dict(error_dict):
    data_str = json.dumps(error_dict, ensure_ascii=False)
    error_log_dir = './error_log.txt'
    with open(error_log_dir, "a", encoding='utf-8') as f:
        f.write(data_str + "\n")

if __name__ == '__main__':
    lr_list = [1E-3, 1E-4, 1E-5]
    bs_list = [32, 64, 128]
    wd_list = [1E-2, 1E-3, 1E-4]
    opt_list = ['sgd', 'adam']
    drop_list = [0.3, 0.5, 0.7]
    hid_list = [16, 32, 64]
    end_list = [32, 16, 128]
    model_list = [1]




    # for lr in lr_list:
    #     for opt in opt_list:
    #         for model in model_list:
    #             for mode in mode_list:
    #                 command_run([' --lr', str(lr), ' --optimizer', opt, ' --modelname', str(model), ' --SubTestFlag', mode])

    """
   running baseline 
    """
    subs = 15
    model_list = ['DGCNN', 'TSCeption', 'DCATT', 'EmT']
    sys_time = get_sys_time()

    for model in model_list:
        mode = 'INS'
        bs = 64
        opt = 'adam'
        result_dir = './train_result_baseline_INS/'
        lr, dataset = model_init(model)

        for sub in range(subs):
            command = [' --lr', str(lr), ' --batch_size', str(bs), '--optimizer', opt,
                       ' --dataset', dataset, ' --SubTestFlag', mode, ' --subject', str(sub),
                       ' --modelname', model, ' --root_dir', result_dir]
            command_run(command, sys_time)

    for model in model_list:
        mode = 'LOSO'
        bs = 64
        opt = 'adam'
        result_dir = './train_result_baseline_LOSO/'
        lr, dataset = model_init(model)

        for sub in range(subs):
            command = [' --lr', str(lr), ' --batch_size', str(bs), '--optimizer', opt,
                       ' --dataset', dataset, ' --SubTestFlag', mode, ' --subject', str(sub),
                       ' --modelname', model, ' --root_dir', result_dir]
            command_run(command, sys_time)





