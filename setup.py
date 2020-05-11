"""The setup script"""

from setuptools import setup
import fdet

setup(
    name='fdet',
    version=fdet.__version__,
    author=fdet.__author__,
    author_email=fdet.__email__,
    maintainer=fdet.__author__,
    maintainer_email=fdet.__email__,
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
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',  # This is important!
    keywords='face recognition detection biometry',
    packages=['fdet'],
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[line for line in open('requirements.txt').read().split('\n') if line != ''],
    entry_points={
        'console_scripts': ['fdet=fdet.cli.fdet:main']
    }
)
