"""
Setup script for Customer Churn Analysis package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='customer-churn-analysis',
    version='1.0.0',
    author='Vaibhav Hirpara',
    author_email='vaibhavhirpara095@example.com',
    description='Customer Churn Analysis and Prediction using Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vaibhavhirpara095/Customer-Churn-Analysis',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'churn-analysis=main:main',
        ],
    },
    include_package_data=True,
    keywords='machine-learning churn-prediction customer-analytics data-science',
)
