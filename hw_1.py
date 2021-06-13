def load_dataset(dataset_dir, dataset_name, is_log=False):
    data = []
    flag = []
    with open(dataset_dir + dataset_name, 'r') as dataset:
        for data_line in dataset.readlines():
            data_line = data_line[:-2]
            data_line = data_line.split(' ')
            data.append(data_line[:-1])
            flag.append(data_line[-1])
            if is_log:
                print(f"{dataset_name}: class: {data_line[-1]}, len: {len(data_line)}, data: {data_line}")
    return data, flag


def TreeGenerate(D, A):
    node = None


if __name__ == '__main__':
    dataset_train, flag_train = load_dataset(dataset_dir='./dataset/',
                                             dataset_name='dna.data',
                                             is_log=False)
    dataset_test, flag_test = load_dataset(dataset_dir='./dataset/',
                                           dataset_name='dna.test',
                                           is_log=False)
    print(len(dataset_train[0]))
    print(len(flag_train))
