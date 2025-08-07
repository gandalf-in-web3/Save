"""
安装脚本
"""

from setuptools import setup, find_packages


setup(
    name="tidemodel",
    version="0.0.0",
    description="model part of tidequant",
    author="wangwenxi",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
)
