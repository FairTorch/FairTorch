from setuptools import setup

setup(
    name='FairTorch',
    version='0.1.0',
    description='A library for fair machine learning created for the PyTorch Summer Hackathon 2020.',
    url='https://github.com/FairTorch/FairTorch',
    author='FairTorch',
    author_email='fairtorch@gmail.com',
    license='BSD 2-clause',
    packages=['FairTorch'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
