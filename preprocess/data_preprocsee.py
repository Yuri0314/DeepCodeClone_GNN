import os

from tqdm import tqdm


def generate_pair_data(dataset_name):
    """
    只用于处理googlejam4_src数据集
    """
    if os.path.exists(os.path.join('.', dataset_name + '_true_clone_pairs.dat')) \
            and os.path.exists(os.path.join('.', dataset_name + '_false_clone_pairs.dat')):
        print(dataset_name + ' clone pair data file exist!!!')
        return
    true_clone_pairs = []
    false_clone_pairs = []
    dataset_dir = os.path.join('./DataSet', dataset_name)
    dirs = os.listdir(dataset_dir)
    for i in range(len(dirs)):
        dir1 = dirs[i]
        curDir = os.path.join(dataset_dir, dir1)
        files = os.listdir(curDir)
        # Add true clone pairs
        for fi in range(len(files)):
            for fj in range(fi + 1, len(files)):
                true_clone_pairs.append((os.path.join(curDir, files[fi]), os.path.join(curDir, files[fj]), '1'))
        # Add false clone pairs
        for j in range(i + 1, len(dirs)):
            dir2 = dirs[j]
            anotherDir = os.path.join(dataset_dir, dir2)
            otherFiles = os.listdir(anotherDir)
            for f1 in files:
                for f2 in otherFiles:
                    false_clone_pairs.append((os.path.join(curDir, f1), os.path.join(anotherDir, f2), '0'))
    with open(os.path.join('.', dataset_name + '_true_clone_pairs.dat'), 'w') as f:
        for pair in true_clone_pairs:
            f.write(pair[0] + ' ' + pair[1] + ' ' + pair[2] + '\n')
    with open(os.path.join('.', dataset_name + '_false_clone_pairs.dat'), 'w') as f:
        for pair in false_clone_pairs:
            f.write(pair[0] + ' ' + pair[1] + ' ' + pair[2] + '\n')

    print(len(true_clone_pairs))
    print(len(false_clone_pairs))
    print('Generate pair data file done...', flush=True)


def split_true_false_data(dataset_name):
    """
    只用于处理bigclonebenchdata数据集
    """
    if os.path.exists(os.path.join('.', dataset_name + '_true_clone_pairs.dat')) \
            and os.path.exists(os.path.join('.', dataset_name + '_false_clone_pairs.dat')):
        print(dataset_name + ' clone pair data file exist!!!')
        return
    true_clone_pairs = []
    false_clone_pairs = []
    pair_data_file = os.path.join('./DataSet', dataset_name + '.dat')
    with open(pair_data_file, 'r') as f:
        for line in tqdm(f.readlines(), desc='Preprocess {} file'.format(dataset_name)):
            tmp_list = line.split()
            tmp_list[0] = './DataSet' + tmp_list[0][1:]
            tmp_list[1] = './DataSet' + tmp_list[1][1:]
            if tmp_list[2] == str(1):
                true_clone_pairs.append(tmp_list)
            else:
                false_clone_pairs.append(tmp_list)

    with open(os.path.join('.', dataset_name + '_true_clone_pairs.dat'), 'w') as f:
        for pair in tqdm(true_clone_pairs, desc='Writing true clone pair data'):
            f.write(pair[0] + ' ' + pair[1] + ' ' + pair[2] + '\n')
    with open(os.path.join('.', dataset_name + '_false_clone_pairs.dat'), 'w') as f:
        for pair in tqdm(false_clone_pairs, desc='Writing false clone pair data'):
            f.write(pair[0] + ' ' + pair[1] + ' ' + pair[2] + '\n')

    print(len(true_clone_pairs))
    print(len(false_clone_pairs))


if __name__ == '__main__':
    dataset_name = 'googlejam4_src'
    generate_pair_data(dataset_name)
