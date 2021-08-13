import os


def remove_files_by_name(starts_with='noise', directory='..'):
    # print(os.listdir(directory))
    # print('starts with: ', starts_with)
    files = [f for f in os.listdir(directory) if
             os.path.isfile(f) and f.startswith(starts_with)]
    for f in files:
        print('remove file: ', f)
        os.remove(f)


if __name__ == "__main__":
    remove_files_by_name(starts_with='noise')
