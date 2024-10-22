import argparse

def convert_to_Bytes(nccl_msg_size_number,nccl_msg_size_unit):
    if nccl_msg_size_unit == 'G':
        return 1024*1024*1024*int(nccl_msg_size_number)
    if nccl_msg_size_unit == 'M':
        return 1024*1024*int(nccl_msg_size_number)
    if nccl_msg_size_unit == 'K':
        return 1024*int(nccl_msg_size_number)
    if nccl_msg_size_unit == 'B':
        return int(nccl_msg_size_number)

def check_for_bytes(nccl_message_size):
    # Check if input is in Bytes
    if nccl_message_size[-1] not in ['K','M','G']:
        return(True)


def generate_nccl_msg_list(nccl_message_size_begin,nccl_message_size_end):

    if check_for_bytes(nccl_message_size_begin):
        nccl_message_size_begin_number = nccl_message_size_begin
        nccl_message_size_begin_unit = 'B'
    else:
        nccl_message_size_begin_number = nccl_message_size_begin[0:-1]
        nccl_message_size_begin_unit = nccl_message_size_begin[-1]

    if check_for_bytes(nccl_message_size_end):
        nccl_message_size_end_number = nccl_message_size_end
        nccl_message_size_end_unit = 'B'
    else:
        nccl_message_size_end_number = nccl_message_size_end[0:-1]
        nccl_message_size_end_unit = nccl_message_size_end[-1]


    nccl_message_size_begin_number_Bytes = convert_to_Bytes(nccl_message_size_begin_number,
                                                  nccl_message_size_begin_unit)
    nccl_message_size_end_number_Bytes = convert_to_Bytes(nccl_message_size_end_number,
                                                  nccl_message_size_end_unit)

    nccl_msg_size = nccl_message_size_begin_number_Bytes
    nccl_msg_size_list = []
    while nccl_msg_size <= nccl_message_size_end_number_Bytes:
        nccl_msg_size_list.append(str(nccl_msg_size))
        nccl_msg_size = nccl_msg_size * 2

    return(nccl_msg_size_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My Script')
    parser.add_argument('nccl_message_size_begin', type=str, help='Beginning NCCL Msg Size')
    parser.add_argument('nccl_message_size_end', type=str, help='Ending NCCL Msg Size')
    args = parser.parse_args()

    nccl_message_size_begin = args.nccl_message_size_begin
    nccl_message_size_end = args.nccl_message_size_end

    nccl_msg_size_list = generate_nccl_msg_list(nccl_message_size_begin,nccl_message_size_end)

    print(" ".join(nccl_msg_size_list))
