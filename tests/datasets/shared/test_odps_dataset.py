# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.shared.odps_reader import (OdpsReader,
                                                set_dataloader_workid,
                                                set_dataloader_worknum)

if __name__ == '__main__':
    print('test for odps reader')

    odps_config = {
        'access_id': 'LTAxxxxxxxxxxxqfMujM4',
        'access_key': 'cqxxxxxxxxxxxUvRyFV',
        'end_point': 'http://service-corp.odps.aliyun-inc.com/api'
    }

    table_name = 'odps://sre_mpi_algo_dev/tables/tb_fpage_100_image_id'

    set_dataloader_worknum(10)
    set_dataloader_workid(3)

    a = OdpsReader(
        table_name=table_name,
        odps_io_config=odps_config,
        image_col=['main_url'],
        image_type=['url'])
    # a = OdpsReader(table_name=table_name, odps_config=odps_config, random_start=False)

    print(a.get_length())
    # print(a.url_image_idx)
    # print(a.base64_image_idx)
    for i in range(10):
        # if i% 100 == 0 or i % 100 == 1:
        print(i, a.get_sample(i))
