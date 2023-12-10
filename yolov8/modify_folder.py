import os

def process_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified = False  

    with open(file_path, 'w') as file:
        for line in lines:
            elements = line.split(' ')
            
            if elements and elements[0] == '19':
                elements[0] = '0'
                modified = True
            
            file.write('\t'.join(elements))

    if modified:
        print(f"Modified file: {file_path}")
    else:
        print(f"No modification made in file: {file_path}")

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_text_file(file_path)