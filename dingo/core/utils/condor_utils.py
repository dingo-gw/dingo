import os
from os.path import join
import yaml


def resubmit_condor_job(train_dir, train_settings, epoch):
    """
    TODO: documentation
    :param train_dir:
    :param train_settings:
    :param epoch:
    :return:
    """
    if 'condor_settings' in train_settings:
        print('Copying log files')
        copy_logfiles(train_dir, epoch=epoch)

        if epoch >= train_settings['train_settings']['runtime_limits'][
            'max_epochs_total']:
            print('Training complete, job will not be resubmitted')
        else:
            print('Training incomplete, resubmitting job.')
            create_submission_file_and_submit_job(train_dir)


def create_submission_file_and_submit_job(train_dir,
                                          filename='submission_file.sub'):
    """
    TODO: documentation
    :param train_dir:
    :param filename:
    :return:
    """
    create_submission_file(train_dir, filename)
    with open(join(train_dir, 'train_settings.yaml'), 'r') as fp:
        bid = yaml.safe_load(fp)['condor_settings']['bid']
    os.system(f'condor_submit_bid {bid} {join(train_dir, filename)}')


def create_submission_file(train_dir, filename='submission_file.sub'):
    """
    TODO: documentation
    :param train_dir:
    :param filename:
    :return:
    """
    with open(join(train_dir, 'train_settings.yaml'), 'r') as fp:
        d = yaml.safe_load(fp)['condor_settings']
    lines = []
    lines.append(f'executable = {d["python"]}\n')
    lines.append(f'request_cpus = {d["num_cpus"]}\n')
    lines.append(f'request_memory = {d["memory_cpus"]}\n')
    lines.append(f'request_gpus = {d["num_gpus"]}\n')
    lines.append(f'requirements = TARGET.CUDAGlobalMemoryMb > '
                 f'{d["memory_gpus"]}\n\n')

    lines.append(f'arguments = {d["train_script"]} --train_dir {train_dir}\n')
    lines.append(f'error = {join(train_dir, "info.err")}\n')
    lines.append(f'output = {join(train_dir, "info.out")}\n')
    lines.append(f'log = {join(train_dir, "info.log")}\n')
    lines.append('queue')

    with open(join(train_dir, filename), 'w') as f:
        for line in lines:
            f.write(line)


def copyfile(src, dst):
    os.system('cp -p %s %s' % (src, dst))


def copy_logfiles(log_dir, epoch, name='info', suffixes=('.err','.log','.out')):
    for suffix in suffixes:
        src = join(log_dir, name + suffix)
        dest = join(log_dir, name + '_{:03d}'.format(epoch) + suffix)
        try:
            copyfile(src, dest)
        except:
            print('Could not copy ' + src)


if __name__ == '__main__':
    train_dir = '/Users/mdax/Documents/dingo/devel/dingo-devel/tutorials/02_gwpe/train_dir/'
    create_submission_file(train_dir)

    # epoch = pm.epoch - 1
    # if args.checkpoint_frequency is None:
    #     # save model and copy logfiles
    #     print('Saving model to model_{:03d}.pt.'.format(epoch))
    #     pm.save_model(filename='model_{:03d}.pt'.format(epoch))
    # print('Copying logfiles.')
    # copy_logfiles(pm.history_dir, epoch=epoch)
    #
    # # If training is not finished, resubmit the submission file
    # if epoch >= args.epochs_total:
    #     print('Training complete, job will not be resubmitted')
    # elif args.restart:
    #     create_submission_file(args.history_dir, name='submission_file_x.sub')
    #     resubmit_condor_job(
    #         submission_file=join(args.history_dir, 'submission_file_x.sub'),
    #         bid=5)
