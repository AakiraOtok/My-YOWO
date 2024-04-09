import yaml

def load_yaml_file(file_path = './config.yaml'):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


config = load_yaml_file()
a = config['max_epoch']

print(type(a))