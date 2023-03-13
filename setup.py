from setuptools import find_packages,setup
hypen_e_dot = "-e ."

def get_requirements(file_path:str):
    rx = []
    with open(file_path) as file_obj:
        rx =file_obj.readlines()
        rx= [x.replace("\n","") for x in rx]
        if hypen_e_dot in rx:
            rx.remove(hypen_e_dot)
    return rx


setup(
    name="base.setup",
    version="0.0.1",
    author="Siddharth Kamble",
    author_email="Siddharthsk7t@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),

)


