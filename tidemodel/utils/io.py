"""
IO读写
"""

from typing import List


def read_txt(file: str) -> List[str]:
    """
    从一个txt文件路径中读取所有行
    """
    
    with open(file, 'r') as f:
        contents: List[str] = f.readlines()
        contents = [content.strip() for content in contents]
    return contents


def write_txt(file: str, contents: List[str]) -> None:
    """
    将一个列表写入到txt中
    """

    with open(file, 'w') as f:
        for content in contents:
            f.write(content)
            f.write('\n')
