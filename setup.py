from __future__ import print_function
import os
import re
from setuptools import setup, Extension, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess

cd_dir = os.path.abspath(os.path.dirname(__file__))
cwd = os.getcwd()
os.chdir(cd_dir)

def get_requirements(get_links=False):
    """
    Get the required packages for install
    """
    links = []
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    for requirement in requirements:
        if requirement.startswith(("git+", "svn+", "hg+")):
            requirements.remove(requirement)
            requirements.append(requirement.split("#egg=")[1])
            if requirement.startswith("git+"):
                links.append(requirement.split("git+")[1])
            elif requirement.startswith("svn+"):
                links.append(requirement.split("svn+")[1])
            else:
                links.append(requirement.split("hg+")[1])
    if get_links:
        return links
    return requirements

version_file = os.path.join("./countdist2", "_version.py")
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    cd_version = mo.group(1)
    print("CountDist2 version = ", cd_version)
else:
    raise RuntimeError("Unable to find version string in %s" % (version_file,))

try:
    os.link("./run", "./countdist2")
except OSError:
    pass


def compile_code(v=None, omp_num_threads=None, prefix="/usr/local",
        run_install=True):
    if v is not None:
        v_command = "VERS={} ".format(v)
    else:
        v_command = ""
    if omp_num_threads is not None:
        o_command = "OMP={} ".format(omp_num_threads)
    else:
        o_command = ""
    cmake_command = "{}{}cmake -DCMAKE_INSTALL_PREFIX={} .".format(
        v_command, o_command, prefix)
    subprocess.check_call(cmake_command, shell=True)
    subprocess.check_call("make", shell=True)
    if run_install:
        subprocess.check_call("make install", shell=True)
    else:
        os.link(os.path.join(cd_dir, "build", "bin", "run"),
                os.path.join(prefix, "bin", "run"))


class CustomInstall(install):
    user_options = install.user_options + [
        ("omp=", None,
         "Number of threads to use for OpenMP: if None (default) or 0, "
         "OpenMP not used"),
        ("loc=", None,
         "Prefix to use for executable install location, if not /usr/local"),
        ("v=", None,
         "Version of C++ code to use: if not given, will use version 4")]

    def initialize_options(self):
        """Set default values"""
        install.initialize_options(self)
        self.v = None
        self.omp = None
        self.loc = "/usr/local"

    def finalize_options(self):
        """Post-process options"""
        install.finalize_options(self)

    def run(self):
        with open("extras.txt", "w") as f:
            f.write(os.path.join(self.loc, "bin", "run"))
        compile_code(v=self.v, omp_num_threads=self.omp, prefix=self.loc)
        install.run(self)

class CustomDevelop(develop):
    user_options = develop.user_options + [
        ("omp=", None,
         "Number of threads to use for OpenMP: if None (default) or 0, "
         "OpenMP not used"),
        ("loc=", None,
         "Prefix to use for executable install location, if not /usr/local"),
        ("v=", None,
         "Version of C++ code to use: if not given, will use version 4")]

    def initialize_options(self):
        self.v = None
        self.omp = None
        self.loc = "/usr/local"
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        if self.uninstall:
            with open("extras.txt") as f:
                bin_copy = f.read()
            os.unlink(bin_copy)
        else:
            with open("extras.txt", "w") as f:
                f.write(os.path.join(self.loc, "bin", "run"))
                compile_code(v=self.v, omp_num_threads=self.omp,
                        prefix=self.loc, run_install=False)
        develop.run(self)

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
        with open("extras.txt") as f:
            bin_loc = f.read()
        os.remove(bin_loc)


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
             packages=["countdist2"], include_package_data=True,
             install_requires=get_requirements(),
             dependency_links=get_requirements(True), cmdclass={"clean":
                 CleanCommand, "uninstall": UninstallCommand, "install":
                 CustomInstall, "develop": CustomDevelop})

os.chdir(cwd)
