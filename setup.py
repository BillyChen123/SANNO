from setuptools import setup, find_packages

# 读取 requirements.txt 文件
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SANNO',  # 项目名称
    version='0.1.2',  # 版本号
    packages=find_packages(),  # 自动查找所有包
    install_requires=requirements,  # 依赖列表
    description='An automated cell type annotation algorithm for unmatched spatial transcriptomics data',  # 项目描述
    long_description=long_description,  # 详细描述
    long_description_content_type='text/markdown',  # 描述内容类型
    author='Billy Chen',  # 作者
    author_email='your.email@example.com',  # 作者邮箱
    url='https://github.com/yourusername/yourproject',  # 项目主页
    python_requires='>=3.6',  # Python版本要求
)