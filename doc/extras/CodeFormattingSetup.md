# Code Formatting Setup Guide

This guide explains how to set up automatic code formatting for IPPL development using the pre-commit git hook.

## Overview

IPPL uses two automatic formatters:
- **clang-format-21**: For C/C++ source files (`.c`, `.cc`, `.cpp`, `.cxx`, `.h`, `.hpp`, etc.)
- **cmake-format**: For CMake configuration files (`CMakeLists.txt`, `*.cmake`)

The pre-commit hook runs these formatters automatically before each commit, ensuring all committed code adheres to the project's styling conventions.

## 1. Installing clang-format-21

### Ubuntu/Debian

Add the LLVM repository and install clang-format-21:

```bash
# Add LLVM repository (Ubuntu 20.04+, Debian 11+)
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-21 main"
sudo apt update

# Install clang-format-21
sudo apt install clang-format-21
```

Verify the installation:

```bash
clang-format-21 --version
```

Note: If you have `clang-format` installed but it doesn't use the full name `clang-format-21`, then just create a symlink in your `.local` path that redirects `clang-format-21` to your real `clang-format` executable  

### APPLE macOS Tahoe
```bash
brew install llvm@21
clang-format --version                                                                                               
```

### Other Systems

For other operating systems, visit the [LLVM download page](https://releases.llvm.org/) or use your system's package manager to locate `clang-format` version 21 or later.

## 2. Installing cmake-format

cmake-format is a Python package available via pip:

```bash
# Install cmake-format (Python 3.x required)
pip install cmake-format
```

Verify the installation:

```bash
cmake-format --version
```

If you prefer to install it system-wide or in a virtual environment, refer to the [cmake-format documentation](https://cmake-format.readthedocs.io/).

## 3. Setting Up the Pre-Commit Hook

### Basic Setup

The repository includes the git hook script. From the repository root, setup a symlink to the git hook (in the .git/hooks dir):

```bash
cd /path/to/ippl
ln -s -f ../../scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```
(note that the extra ../../ in the path is because git runs the hook from the .git/hooks directory, so the symlink has to be redirected)

Alternatively, create a symbolic link manually directly from the git hooks dir:

```bash
cd /path/to/ippl/.git/hooks
ln -s -f ../../scripts/pre-commit pre-commit
chmod +x pre-commit
```

### Verification

To verify the hook is installed, check that the symlink exists:

```bash
ls -la /path/to/ippl/.git/hooks/pre-commit
```

You should see output similar to:

```
lrwxrwxrwx  1 user group  23 Mar  4 12:00 pre-commit -> ../../scripts/pre-commit
```

## 4. How the Hook Works

When you run `git commit`:

1. The pre-commit hook identifies all staged C/C++ and CMake files
2. For each file, it checks if formatting is needed by comparing the staged content with the formatted version
3. If formatting is needed, the hook:
   - Reports which files need reformatting
   - Displays the command to fix each file
   - Prevents the commit from proceeding
4. Run the suggested command and re-stage the files
5. Retry `git commit`

Note that we include the command-line parameter `-style=file` in all clang-format commands mentioned here, but this is actually the default and skipping it should not cause problems. Providing there is a `.clang-format` file present in the repository root, all should work as expected.

### Example Workflow

```bash
# Make changes and stage them
git add src/myfile.cpp

# Try to commit
git commit -m "Add feature"

# If formatting is needed, you'll see:
# clang-format-21 -style=file -i src/myfile.cpp
# Then re-stage and retry
git add src/myfile.cpp
git commit -m "Add feature"
```

## 5. Bypassing the Hook

If you need to commit without running the hook (not recommended), use:

```bash
git commit --no-verify
```

**Warning**: Code committed without formatting may not adhere to project standards.

## 6. Manual Formatting

To format files without committing:

```bash
# Format a single C/C++ file
clang-format-21 -style=file -i src/myfile.cpp

# Format all C/C++ files in a directory
find src -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format-21 -style=file -i {} \;

# Format CMake files
cmake-format -c /path/to/ippl/.cmake-format.py -i CMakeLists.txt
```

## 7. Troubleshooting

### Hook not executing?

- Verify the script is executable: `chmod +x .git/hooks/pre-commit`
- Verify the symlink points to the correct location: `ls -la .git/hooks/pre-commit`
- Ensure `clang-format-21` and `cmake-format` are in your `$PATH`: `which clang-format-21 cmake-format`

### Formatting succeeds locally but hook still rejects?

This is typically due to:
- Different `clang-format-21` versions installed (ensure you have version 21)
- Nested `.clang-format` configuration files in subdirectories that override the root config

### Path issues on HPC systems?

If you're on a shared HPC system where multiple compiler modules are available:
- Load the correct module before working: `module load llvm/21` or similar
- Create a symlink to `clang-format` in `~/.local/bin` if module loading is inconvenient

## 8. Configuration Files

The formatting behavior is controlled by:

- **C/C++ formatting**: `.clang-format` (repository root and subdirectories)
- **CMake formatting**: `.cmake-format.py` (repository root)

These files define all style rules. For detailed customization, refer to:
- [clang-format documentation](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [cmake-format documentation](https://cmake-format.readthedocs.io/)

## Questions?

Refer to the main [WORKFLOW.md](../WORKFLOW.md) guide for additional context on IPPL's development practices.
