"""The setup script"""

from setuptools import setup

AUTHOR = 'Antonio C. Nazare Jr.'
EMAIL = 'antonio.nazare@dcc.ufmg.br'
VERSION = '0.1.0'

setup(
    name='fdet',
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    url='http://github.com/acnazarejr/fdet',
    download_url='http://github.com/acnazarejr/fdet',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Other Audience',
        'Intended Audience :: System Administrators',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 8',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries'
    ],
    description='The fdet is an easy to use face detection implementation based on PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important!
    keywords='face recognition detection biometry',
    packages=['fdet'],
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=[line for line in open('requirements.txt').read().split('\n') if line != ''],
    entry_points={
        'console_scripts': ['fdet=fdet.cli.main:main']
    }
)
