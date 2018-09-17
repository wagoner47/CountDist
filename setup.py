from __future__ import print_function
import os, sys, re, glob
import platform, sysconfig
from shutil import copyfile, copymode
from setuptools import setup, Extension, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion
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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("omp=", None, "Number of OpenMP threads to use, or 0 for no OpenMP"),
        ("vers=", None, "Version of C++ code to compile")
        ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.omp = None
        self.vers = None

    def finalize_options(self):
        build_ext.finalize_options(self)
    
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]
        if self.omp is not None:
            cmake_args += ['-Domp=' + self.omp]
        if self.vers is not None:
            cmake_args += ['-Dvers=' + self.vers]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(os.getenv('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        lib_file = glob.glob(os.path.join(extdir, "calculate_distances*.so"))[0]
        try:
            os.link(lib_file, os.path.join(cd_dir, "countdist2", "calculate_distances.so"))
        except:
            pass

        test_bin = os.path.join(self.build_temp, "CountDistCPP_test{}".format(".exe" if platform.system() == "Windows" else ""))
        self.copy_test_file(test_bin)

    def copy_test_file(self, src_file):
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests", "bin")
        if dest_dir != "":
            os.makedirs(dest_dir, exist_ok=True)

        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        copyfile(src_file, dest_file)
        copymode(src_file, dest_file)


class CatchTestCommand(TestCommand):
    user_options = TestCommand.user_options + [
        ("catch-args=", None, "Arguments to pass to Catch2 for C++ testing")
        ]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.catch_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        if self.catch_args is None:
            self.catch_args = ""
        
    def distutils_dir_name(self, dname):
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        print("\nRunning Python tests...\n")
        TestCommand.run(self)
        print("\nRunning C++ tests...\n")
        subprocess.call('./bin/*_test {}'.format(self.catch_args), cwd=os.path.join(cd_dir, 'tests'), shell=True)

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
             dependency_links=get_requirements(True),
             test_suite="tests",
             ext_modules=[CMakeExtension("calculate_distances")],
             cmdclass={"clean": CleanCommand, "uninstall": UninstallCommand,
                       "build_ext": CMakeBuild, "test": CatchTestCommand})

os.chdir(cwd)
