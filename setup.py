from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='adaptive',
      version='0.1',
      description="""Simulation of the Adaptive Voter Model""",
      long_description=readme(),
      classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research'
      ],
      keywords='simulation voter-model',
      project_urls={
        'Source': 'https://github.com/empiricalstateofmind/adaptive',
        'Tracker': 'https://github.com/empiricalstateofmind/adaptive/issues',
      },
      author='Andrew Mellor',
      author_email='mellor91@hotmail.co.uk',
      license='Apache Software License',
      packages=['adaptive'],
      python_requires='>=3',
      install_requires=[
          'networkx',
          'numpy',
          'matplotlib',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
