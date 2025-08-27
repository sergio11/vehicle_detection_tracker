from setuptools import setup, find_packages

setup(
    name='VehicleDetectionTracker',  
    version='0.0.32',
    packages=find_packages(),
    include_package_data=True,
    package_data={'VehicleDetectionTracker': ['data/*']},
    install_requires=[
        'opencv-python==4.11.0.86',
        'imutils==0.5.4',
        'ultralytics==8.3.87',
        'tensorflow==2.18.0'
    ],
    author='Sergio Sánchez Sánchez',
    author_email='dreamsoftware92@gmail.com',
    description='VehicleDetectionTracker 🚗: Effortlessly track and detect vehicles in images and videos with advanced algorithms. 🚙🚕 Boost your computer vision project!" 🔍📹',
    url='https://github.com/sergio11/vehicle_detection_tracker',
    keywords=['Vehicle tracking', 'Object detection', 'Computer vision', 'Video processing', 'Traffic analysis', 'Traffic management', 'Automated surveillance', 'Vehicle recognition', 'Video analysis', 'Machine learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.7, <4',
    long_description = open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)

"""
VehicleDetectionTracker Setup

This setup script configures the installation of the VehicleDetectionTracker package. VehicleDetectionTracker is a Python package for effortless vehicle tracking and detection in images and videos using advanced algorithms. Boost your computer vision project with this package.

Project Details:
- Name: VehicleDetectionTracker
- Version: 0.0.32
- Author: Sergio Sánchez Sánchez
- Email: dreamsoftware92@gmail.com
- Description: VehicleDetectionTracker is a package that enables effortless tracking and detection of vehicles in images and videos using advanced algorithms. Ideal for enhancing your computer vision project with vehicle recognition and tracking.
- Repository: https://github.com/sergio11/vehicle_detection_tracker
- Keywords: Vehicle tracking, Object detection, Computer vision, Video processing, Traffic analysis, Traffic management, Automated surveillance, Vehicle recognition, Video analysis, Machine learning

Requirements:
- opencv-python  4.11.0.86
- imutils 0.5.4
- ultralytics 8.3.87
- tensorflow 2.18.0

Development Status: Beta

License: MIT License

Python Version Compatibility: >=3.7, <4

For more details, please refer to the project's README.md file.
"""