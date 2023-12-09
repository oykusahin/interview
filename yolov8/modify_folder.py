import os

def process_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified = False  # Flag to track if any modifications were made

    with open(file_path, 'w') as file:
        for line in lines:
            # Splitting the line based on tabs
            elements = line.split(' ')
            
            # Checking if the first element is '19' and replacing it with '0'
            if elements and elements[0] == '19':
                elements[0] = '0'
                modified = True
            
            # Writing the modified or unmodified line back to the file
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