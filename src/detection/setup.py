from setuptools import find_packages, setup
 
package_name = 'detection'
 
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    package_data={'detection': ['best.pt','nmist_model_new.pth']},  # includes the model file
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['srv/TakePhoto.srv']),
    ],
    install_requires=['setuptools', 'roboflow', 'rclpy', 'rosidl_runtime_py'],  # remove 'setuptools' if ROS handles it for you
    zip_safe=True,
    maintainer='team14',
    maintainer_email='harrytolcher1@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],  # optional, if you're not using pytest you can remove this
    entry_points={
        'console_scripts': [
            'detection = detection.detection_nomodel_nn:main'
        ],
    },
)