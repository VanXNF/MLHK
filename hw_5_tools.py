# -*- coding: utf-8 -*-
def write_list_to_txt(data_list, file_path):
    txt_log = open(file_path, 'w')
    for item in data_list:
        txt_log.write(str(item))
        txt_log.write('\n')
    txt_log.close()
