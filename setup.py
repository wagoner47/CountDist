from __future__ import print_function
import os, sys, re, glob
import platform, sysconfig
import pathlib
import shutil
from setuptools import setup, Extension, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion
import subprocess
from configobj import ConfigObj

cd_dir = pathlib.Path(__file__).parent.resolve()
cwd = pathlib.Path.cwd()
os.chdir(cd_dir)

def get_requirements(get_links=False):
    """
    Get the required packages for install
    """
    links = []
    requirements = cd_dir.joinpath("requirements.txt").read_text().splitlines()
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

version_file = cd_dir.joinpath("countdist2", "_version.py")
verstrline = version_file.read_text()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    cd_version = mo.group(1)
    print("CountDist2 version = ", cd_version)
else:
    raise RuntimeError("Unable to find version string in {}".format(
        version_file.relative_to(cd_dir.parent)))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = pathlib.Path(sourcedir).resolve()


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("omp=", None, "Number of OpenMP threads to use, or 0 for no OpenMP"),
        ("vers=", None, "Version of C++ code to compile"),
        ("c=", None, "C compiler to use (blank for system default)"),
        ("cxx=", None, "C++ compiler to use (blank for system default)") 
        ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.omp = None
        self.vers = None
        self.c = None
        self.cxx = None

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following"
                               " extensions: " + ", ".join(e.name for e in
                                                           self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir.parent.resolve()),
            "-DPYTHON_EXECUTABLE=" + sys.executable]
        if self.omp is not None:
            cmake_args += ['-Domp=' + self.omp]
        if self.vers is not None:
            cmake_args += ['-Dvers=' + self.vers]
        if self.c is not None:
            cmake_args += ['-DCMAKE_C_COMPILER=' + self.c]
        if self.cxx is not None:
            cmake_args += ['-DCMAKE_CXX_COMPILER=' + self.cxx]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), str(extdir.parent.resolve()))]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            os.getenv('CXXFLAGS', ''), self.distribution.get_version())

        subprocess.check_call(
            ['cmake', str(ext.sourcedir)] + cmake_args, cwd=build_temp, env=env)
        if not self.dry_run:
            subprocess.check_call(
                ['cmake', '--build', '.'] + build_args, cwd=build_temp)
            try:
                os.link(
                    extdir,
                    cd_dir.joinpath("countdist2", "calculate_distances.so"))
            except:
                pass

        test_bin = build_temp.joinpath("CountDistCPP_test{}".format(
            ".exe" if platform.system() == "Windows" else ""))
        self.copy_test_file(test_bin)

    def copy_test_file(self, src_file):
        dest_dir = cd_dir.joinpath("tests", "bin")
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_file = pathlib.Path(src_file)
        dest_file = dest_dir.joinpath(src_file.name)
        shutil.copyfile(src_file, dest_file)
        shutil.copymode(src_file, dest_file)


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
        print("\nPreparing for Python tests...\n")
        test_data_dir = cd_dir.joinpath("tests", "test_data")
        config = ConfigObj(
            os.fspath(
                test_data_dir.joinpath("params_catalog_three_objects.ini")))
        ifname1 = pathlib.Path(config["run_params"]["ifname1"]).name
        ifname2 = pathlib.Path(config["run_params"]["ifname2"]).name
        cosmo_file = pathlib.Path(config["run_params"]["cosmo_file"]).name
        config["run_params"]["ifname1"] = str(test_data_dir.joinpath(ifname1))
        config["run_params"]["ifname2"] = str(test_data_dir.joinpath(ifname2))
        config["run_params"]["cosmo_file"] = str(
            test_data_dir.joinpath(cosmo_file))
        config.write()
        print("\nRunning Python tests...\n")
        TestCommand.run(self)
        print("\nRunning C++ tests...\n")
        subprocess.check_call(
            [str(pathlib.Path("bin", "*_test")), self.catch_args],
            cwd=cd_dir.joinpath("tests"), shell=True)


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree(cd_dir.joinpath("build"), ignore_errors=True)
        shutil.rmtree(cd_dir.joinpath("dist"), ignore_errors=True)
        [p.unlink() for p in cd_dir.rglob("*.pyc")]
        [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
         for p in cd_dir.rglob("*.tgz")]
        [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
         for p in cd_dir.rglob("*.egg*")]
        [p.unlink() for p in cd_dir.rglob("cmake_install.cmake")]
        [p.unlink() for p in cd_dir.rglob("Makefile")]
        [shutil.rmtree(p, ignore_errors=True) for p in
         cd_dir.rglob("CMakeFiles")]
        [p.unlink() for p in cd_dir.rglob("calculate_distances*.so")]


dist = setup(
    name="CountDist2",
    version=cd_version,
    author="Erika Wagoner",
    author_email="wagoner47@email.arizona.edu",
    description="Get separations between objects in catalogs",
    packages=["countdist2"],
    include_package_data=True,
    install_requires=get_requirements(),
    dependency_links=get_requirements(True),
    test_suite="tests",
    ext_modules=[CMakeExtension("calculate_distances")],
    cmdclass={
        "clean": CleanCommand,
        "build_ext": CMakeBuild,
        "test": CatchTestCommand})

os.chdir(cwd)
