#!/usr/bin/env python3
"""
Cross-platform build helper for GLSL plugins (Linux and macOS only)
"""

import os
import sys
import platform
import subprocess
import argparse

def get_platform():
    """Get the current platform"""
    system = platform.system()
    if system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    else:
        print(f"Error: Unsupported platform '{system}'. Only Linux and macOS are supported.")
        sys.exit(1)

def get_expected_extension():
    """Get the expected library extension for the current platform"""
    if get_platform() == "linux":
        return ".so"
    else:  # macOS
        return ".dylib"

def build_plugin(plugin_dir, build_type="Release"):
    """Build the plugin using CMake"""
    if not os.path.exists(plugin_dir):
        print(f"Error: Plugin directory '{plugin_dir}' does not exist")
        return False
    
    current_platform = get_platform()
    expected_ext = get_expected_extension()
    
    print(f"Building plugin on {current_platform} (expecting {expected_ext} output)")
    
    # Create build directory
    build_dir = os.path.join(plugin_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        # Run CMake configure
        print("Configuring with CMake...")
        subprocess.run([
            "cmake", "..", 
            f"-DCMAKE_BUILD_TYPE={build_type}"
        ], cwd=build_dir, check=True)
        
        # Run build
        print("Building...")
        subprocess.run([
            "cmake", "--build", ".", 
            "--config", build_type,
            "--", "-j4"
        ], cwd=build_dir, check=True)
        
        # Check for output file
        built_files = []
        for file in os.listdir(build_dir):
            if file.endswith(expected_ext):
                built_files.append(file)
        
        if built_files:
            print(f"Build successful! Generated libraries:")
            for file in built_files:
                print(f"  - {file}")
        else:
            print(f"Warning: No {expected_ext} files found in build directory")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: CMake not found. Please install CMake.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build GLSL plugins for Linux and macOS")
    parser.add_argument("plugin_dir", help="Path to plugin directory")
    parser.add_argument("--build-type", choices=["Debug", "Release"], default="Release",
                       help="Build type (default: Release)")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    
    args = parser.parse_args()
    
    # Clean if requested
    if args.clean:
        build_dir = os.path.join(args.plugin_dir, "build")
        if os.path.exists(build_dir):
            print(f"Cleaning build directory: {build_dir}")
            import shutil
            shutil.rmtree(build_dir)
    
    # Build the plugin
    if build_plugin(args.plugin_dir, args.build_type):
        print(f"Plugin build completed successfully on {get_platform()}")
    else:
        print("Plugin build failed")
        sys.exit(1)

if __name__ == "__main__":
    main()