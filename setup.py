from setuptools import setup, find_packages

setup(
    name='securityGPT',      
    version='0.0.1',         
    author='Alexy Skoutnev, Zoe Luther',
    author_email='alexy.a.skoutnev@vanderbilt.edu, zofia.m.luther@vanderbilt.edu',  # Replace with your email
    description='Custom LLM model to classify bug report documents',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages()
)