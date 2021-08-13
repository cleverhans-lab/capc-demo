from utils.remove_files import remove_files_by_name


def init_log_files():
    names = ['server', 'client', 'privacy_guardian']
    prefix = 'logs'
    for name in names:
        file_name = f"{name}_log.txt"
        remove_files_by_name(starts_with=file_name, directory=f'{prefix}')
        with open(file=f"{prefix}/{file_name}", mode='a') as f:
            print('create file: ', file_name)
            f.write(f"init log for: {name}\n")


if __name__ == "__main__":
    init_log_files()
