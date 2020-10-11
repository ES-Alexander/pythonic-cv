import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pythonic-cv',
    version='1.1.4',
    author='ES-Alexander',
    author_email='sandman.esalexander@gmail.com',
    description='Performant pythonic wrapper of unnecessarily painful opencv functionality',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ES-Alexander/pythonic-cv',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
