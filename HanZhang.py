import os
import glob

def remove_last_two_chars_from_filenames(directory, file_extension='.png'):
    """
    遍历指定目录中的所有文件，并将每个文件名的最后两位字符去掉，然后保存为新的文件名。
    
    :param directory: 要处理的目录路径
    :param file_extension: 文件扩展名，默认为 '.png'
    """
    # 获取目录中所有匹配的文件路径
    file_paths = glob.glob(os.path.join(directory, '*' + file_extension))
    
    for file_path in file_paths:
        # 提取文件名和目录路径
        file_name = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        
        # 去掉文件名的最后两位字符（不包括扩展名）
        new_file_name = file_name[:-6] + file_extension  # 假设扩展名长度为4，去掉最后两位字符
        new_file_path = os.path.join(dir_path, new_file_name)
        
        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file_path} -> {new_file_path}")

# 示例用法
if __name__ == '__main__':
    directory = r"C:\Users\zh321\Github\simple_real_time_super_resolution\dataset\SR\RLSR\DIV2K_X2\val_LR"
    remove_last_two_chars_from_filenames(directory)