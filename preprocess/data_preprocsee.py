import os


def generate_pair_data(dataset_name):
    true_clone_pairs = []
    false_clone_pairs = []
    dataset_dir = os.path.join('./DataSet', dataset_name)
    dirs = os.listdir(dataset_dir)
    for i in range(len(dirs)):
        dir1 = dirs[i]
        curDir = os.path.join(dataset_dir, dir1)
        files = os.listdir(curDir)
        print(len(files))
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
    print('Generate pair data done...')


if __name__ == '__main__':
    dataset_name = 'googlejam4_src'
    generate_pair_data(dataset_name)
