from __future__ import print_function
import os
import re
from setuptools import setup, Extension, Command
from setuptools.command.install import install
import subprocess

cd_dir = os.path.abspath(os.path.dirname(__file__))
cwd = os.getcwd()
os.chdir(cd_dir)

with open("requirements.txt") as f:
    required = f.read().splitlines()


version_file = os.path.join("./countdist2", "_version.py")
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    cd_version = mo.group(1)
    print("CountDist2 version = ", cd_version)
else:
    raise RuntimeError("Unable to find version string in %s" %(version_file,))


try:
    os.link("./run", "./countdist2")
except OSError:
    pass


def compile_code(omp_num_threads=None):
    if omp_num_threads is not None:
        command = "OMP={} cmake .".format(omp_num_threads)
    else:
        command = "cmake ."
    subprocess.check_call(command, shell=True)
    subprocess.check_call("make", shell=True)
    subprocess.check_call("make install", shell=True)


class CustomInstall(install):
    user_options = install.user_options + [("omp=", None, "Number of threads to use for OpenMP: "\
            "if None (default) or 0, OpenMP not used")]

    def initialize_options(self):
        """Set default values"""
        install.initialize_options(self)
        self.omp = None

    def finalize_options(self):
        """Post-process options"""
        install.finalize_options(self)

    def run(self):
        compile_code(omp_num_threads=self.omp)
        install.run(self)


class UninstallCommand(Command):
    user_options = [("sudo=", None, "Password for sudo, if any")]
    
    def initialize_options(self):
        """Set default values"""
        self.sudo = None
    
    def finalize_options(self):
        """Post-process options"""
        self.use_sudo = (self.sudo is not None)
    
    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")
        if self.use_sudo:
            command = "{echo {}; echo y;} | sudo pip uninstall " \
                      "countdist2".format(
                self.sudo)
        else:
            command = "echo y | pip uninstall countdist2"
        os.system(command)
        os.system("make uninstall")


class CleanCommand(Command):
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")
        os.system("make clean")


dist = setup(name="CountDist2", version=cd_version, author="Erika Wagoner",
        author_email="wagoner47@email.arizona.edu", description="Get "\
                "separations between objects in catalogs",
                packages=["countdist2"],
                scripts=["build/bin/run"],
                install_requires=required, cmdclass={"clean":CleanCommand,
                    "uninstall":UninstallCommand, "install":CustomInstall})
# dist = setup(name="CountDist2", version=cd_version, author="Erika Wagoner",
#              author_email="wagoner47@email.arizona.edu",
#              description="Get separations between objects in catalogs",
#              packages=["countdist2"], package_data={"countdist2":headers},
#              ext_modules=[ext], install_requires=required,
#              cmdclass={"clean":CleanCommand, "uninstall":UninstallCommand})

# build_lib = glob.glob(os.path.join("build", "*", "countdist2", "_countdist2*.so"))
# if len(build_lib) >= 1:
#     lib = os.path.join("countdist2", "_countdist2.so")
#     if os.path.lexists(lib):
#         os.unlink(lib)
#     os.link(build_lib[0], lib)

os.chdir(cwd)
