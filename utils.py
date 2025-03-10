def log(text, label='', filename='out.log'):
    with open(filename, 'a') as file:
        file.write('\n\n' + label + '\n\n' if label else '')
        file.write(text)


def clear_logs(filenames):
    for filename in filenames:
        try:
            with open(filename, 'w'):
                pass
            print(f'cleared [{filename}]')
        except:
            print(f'failure [{filename}]')