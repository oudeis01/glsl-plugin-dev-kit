#!/usr/bin/env python3
"""
Generic GLSL Plugin Generator
Extracts function metadata from any GLSL shader library and generates C++ headers
"""

import os
import re
import glob
import sys
import argparse
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FunctionSignature:
    return_type: str
    param_types: List[str]

@dataclass
class GLSLFunction:
    name: str
    file_path: str  # relative to the directory containing the .glsl files
    overloads: List[FunctionSignature]

@dataclass
class PluginInfo:
    name: str
    version: str = "0.0.1"
    author: str = "Unknown"

class GLSLParser:
    def __init__(self, plugin_name: str, input_root: str, output_dir: str, version: str = "0.0.1", author: str = "Unknown"):
        self.input_root = Path(input_root)
        self.output_dir = Path(output_dir)
        self.plugin_name = plugin_name
        self.version = version
        self.author = author
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.functions: Dict[str, GLSLFunction] = {}
        self.processed_files: Set[str] = set()
        
        # GLSL type patterns
        self.glsl_types = {
            'float', 'vec2', 'vec3', 'vec4', 
            'mat2', 'mat3', 'mat4',
            'int', 'ivec2', 'ivec3', 'ivec4',
            'bool', 'bvec2', 'bvec3', 'bvec4',
            'sampler2D', 'samplerCube', 'sampler3D',
            'QUAT'  # custom types
        }
        
    def find_glsl_files(self) -> List[Path]:
        """Find all .glsl files in the input directory"""
        glsl_files = []
        for file_path in self.input_root.rglob("*.glsl"):
            glsl_files.append(file_path)
        return sorted(glsl_files)
    
    def remove_includes_and_defines(self, content: str) -> str:
        """Remove #include and #define lines (both guards and macro functions)"""
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip includes
            if stripped.startswith('#include'):
                continue
            # Skip all #define lines (guards and macro functions)
            if stripped.startswith('#define'):
                continue
            # Skip conditional compilation directives
            if stripped.startswith('#ifndef') or stripped.startswith('#ifdef') or stripped.startswith('#endif'):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def extract_use_info(self, content: str) -> List[str]:
        """Extract type information from 'use:' comments"""
        use_patterns = []
        lines = content.split('\n')
        
        for line in lines:
            # Look for use: patterns in comments
            match = re.search(r'use:\s*(.+)', line)
            if match:
                use_patterns.append(match.group(1).strip())
        
        return use_patterns
    
    def parse_function_signature(self, func_line: str) -> Optional[FunctionSignature]:
        """Parse a single function signature line"""
        # Clean up the line
        func_line = func_line.strip()
        if not func_line or func_line.startswith('//'):
            return None
        
        # Remove const, in, out modifiers for parsing
        clean_line = re.sub(r'\b(const|in|out)\b\s+', '', func_line)
        
        # Pattern for function: return_type function_name(params) { ... }
        func_pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{?'
        match = re.match(func_pattern, clean_line)
        
        if not match:
            return None
        
        return_type = match.group(1)
        func_name = match.group(2)
        params_str = match.group(3)
        
        # Only process if return type is a known GLSL type
        if return_type not in self.glsl_types:
            return None
        
        # Parse parameters
        param_types = []
        if params_str.strip():
            params = [p.strip() for p in params_str.split(',')]
            for param in params:
                # Extract type from parameter (ignore variable names)
                param_parts = param.split()
                if param_parts and param_parts[0] in self.glsl_types:
                    param_types.append(param_parts[0])
        
        return FunctionSignature(return_type, param_types)
    
    def parse_macro_function(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse macro function definitions like #define func(args) body"""
        line = line.strip()
        if not line.startswith('#define'):
            return None
        
        # Pattern: #define function_name(args) body
        macro_pattern = r'#define\s+(\w+)\s*\(([^)]*)\)\s+(.+)'
        match = re.match(macro_pattern, line)
        
        if not match:
            return None
        
        func_name = match.group(1)
        params = match.group(2)
        body = match.group(3)
        
        # Skip simple token replacements
        if not params.strip():
            return None
        
        return func_name, params
    
    def extract_functions_from_conditional_block(self, block: str) -> List[Tuple[str, FunctionSignature]]:
        """Extract functions from a conditional compilation block"""
        functions = []
        lines = block.split('\n')
        
        for line in lines:
            # Skip lines that are macro definitions
            if line.strip().startswith('#define'):
                continue
            
            # Try regular function only (exclude macros)
            signature = self.parse_function_signature(line)
            if signature:
                # Extract function name from the line
                func_match = re.search(r'\w+\s+(\w+)\s*\(', line)
                if func_match:
                    func_name = func_match.group(1)
                    functions.append((func_name, signature))
        
        return functions
    
    def parse_conditional_blocks(self, content: str) -> List[str]:
        """Parse conditional compilation blocks and return all possible code paths"""
        blocks = []
        lines = content.split('\n')
        current_block = []
        in_conditional = False
        brace_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Track if we're in a conditional block
            if stripped.startswith(('#ifdef', '#ifndef', '#if ')):
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                in_conditional = True
                i += 1
                continue
            
            if stripped.startswith('#else'):
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                i += 1
                continue
            
            if stripped.startswith('#endif'):
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                in_conditional = False
                i += 1
                continue
            
            # Add line to current block
            current_block.append(line)
            i += 1
        
        # Add remaining block
        if current_block:
            blocks.append('\n'.join(current_block))
        
        # If no conditional blocks found, return the whole content as one block
        if not blocks:
            blocks.append(content)
        
        return blocks
    
    def process_file(self, file_path: Path) -> None:
        """Process a single GLSL file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return
        
        # Get relative path from input root
        rel_path = file_path.relative_to(self.input_root)
        
        # Skip if this is just an include file (only contains includes)
        content_without_comments = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content_without_comments = re.sub(r'//.*', '', content_without_comments)
        
        # Remove includes and guards
        processed_content = self.remove_includes_and_defines(content_without_comments)
        
        # Check if there's actual code left
        code_lines = [line.strip() for line in processed_content.split('\n') 
                     if line.strip() and not line.strip().startswith('#')]
        
        if not code_lines:
            print(f"Skipping {rel_path} - no function definitions found")
            return
        
        print(f"Processing {rel_path}...")
        
        # Parse conditional blocks
        blocks = self.parse_conditional_blocks(processed_content)
        
        # Extract functions from each block
        file_functions = {}
        for block in blocks:
            functions = self.extract_functions_from_conditional_block(block)
            for func_name, signature in functions:
                if func_name not in file_functions:
                    file_functions[func_name] = []
                file_functions[func_name].append(signature)
        
        # Add to global functions dict
        for func_name, signatures in file_functions.items():
            if func_name in self.functions:
                # Merge overloads
                self.functions[func_name].overloads.extend(signatures)
            else:
                self.functions[func_name] = GLSLFunction(
                    name=func_name,
                    file_path=str(rel_path),
                    overloads=signatures
                )
        
        self.processed_files.add(str(rel_path))
    
    def remove_duplicate_overloads(self) -> None:
        """Remove duplicate function overloads"""
        for func in self.functions.values():
            # Create a set to track unique signatures
            unique_sigs = []
            seen = set()
            
            for sig in func.overloads:
                sig_key = (sig.return_type, tuple(sig.param_types))
                if sig_key not in seen:
                    seen.add(sig_key)
                    unique_sigs.append(sig)
            
            func.overloads = unique_sigs
    
    def generate_cpp_header(self, output_filename: str = None) -> str:
        """Generate C++ header file with function metadata"""
        if not output_filename:
            output_filename = f"{self.plugin_name.title()}Plugin.h"
        
        output_path = self.output_dir / output_filename
        plugin_info = PluginInfo(name=self.plugin_name, version=self.version, author=self.author)
        namespace_name = f"{self.plugin_name.title()}Plugin"
        
        with open(output_path, 'w') as f:
            f.write(f"""#pragma once
#include "glsl-plugin-interface/include/GLSLTypes.h"
#include "glsl-plugin-interface/include/IPluginInterface.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace {namespace_name} {{

// Plugin Information
extern const PluginInfo PLUGIN_INFO;

// All Functions (optimized with hardcoded overloads)
extern const std::vector<GLSLFunction> FUNCTIONS;

// Function lookup map (O(1) search)
extern const std::unordered_map<std::string, const GLSLFunction*> FUNCTION_MAP;

// Utility functions
const GLSLFunction* findFunction(const std::string& name);
std::vector<std::string> getAllFunctionNames();

}} // namespace {namespace_name}
""")
        
        return str(output_path)
    
    def generate_cpp_implementation_split(self, functions_per_file: int = 50) -> list:
        """Generate optimized C++ implementation split into multiple files for faster compilation"""
        header_filename = f"{self.plugin_name.title()}Plugin.h"
        plugin_info = PluginInfo(name=self.plugin_name, version=self.version, author=self.author)
        namespace_name = f"{self.plugin_name.title()}Plugin"
        
        sorted_functions = sorted(self.functions.items())
        total_functions = len(sorted_functions)
        
        # Calculate number of files needed
        num_files = (total_functions + functions_per_file - 1) // functions_per_file
        generated_files = []
        
        # Generate main implementation file with plugin info and declarations
        main_output_path = self.output_dir / f"{self.plugin_name.title()}Plugin.cpp"
        with open(main_output_path, 'w') as f:
            f.write(f'#include "{header_filename}"\n#include <algorithm>\n#include <unordered_map>\n\nnamespace {namespace_name} {{\n\n')
            
            # Plugin info
            f.write(f"""const PluginInfo PLUGIN_INFO("{plugin_info.name}", "{plugin_info.version}", "{plugin_info.author}");

""")
            
            # Forward declarations for part functions
            for i in range(num_files):
                f.write(f"std::vector<GLSLFunction> getFunctionsPart{i+1}();\n")
            
            f.write("\n")
            
            # Main FUNCTIONS array that combines all parts
            f.write("const std::vector<GLSLFunction> FUNCTIONS = []() {\n")
            f.write("    std::vector<GLSLFunction> all_functions;\n")
            for i in range(num_files):
                f.write(f"    auto part{i+1} = getFunctionsPart{i+1}();\n")
                f.write(f"    all_functions.insert(all_functions.end(), part{i+1}.begin(), part{i+1}.end());\n")
            f.write("    return all_functions;\n")
            f.write("}();\n\n")
            
            # Hash map for O(1) lookup
            f.write("const std::unordered_map<std::string, const GLSLFunction*> FUNCTION_MAP = []() {\n")
            f.write("    std::unordered_map<std::string, const GLSLFunction*> map;\n")
            f.write("    for (size_t i = 0; i < FUNCTIONS.size(); ++i) {\n")
            f.write("        map[FUNCTIONS[i].name] = &FUNCTIONS[i];\n")
            f.write("    }\n")
            f.write("    return map;\n")
            f.write("}();\n\n")
            
            # Utility functions
            f.write("""const GLSLFunction* findFunction(const std::string& name) {
    auto it = FUNCTION_MAP.find(name);
    return (it != FUNCTION_MAP.end()) ? it->second : nullptr;
}

std::vector<std::string> getAllFunctionNames() {
    std::vector<std::string> names;
    names.reserve(FUNCTIONS.size());
    for (const auto& func : FUNCTIONS) {
        names.push_back(func.name);
    }
    return names;
}

""")
            f.write(f"}} // namespace {namespace_name}\n")
        
        generated_files.append(str(main_output_path))
        
        # Generate part files
        for i in range(num_files):
            start_idx = i * functions_per_file
            end_idx = min((i + 1) * functions_per_file, total_functions)
            part_functions = sorted_functions[start_idx:end_idx]
            
            part_filename = f"{self.plugin_name.title()}Plugin_Part{i+1}.cpp"
            part_output_path = self.output_dir / part_filename
            
            with open(part_output_path, 'w') as f:
                f.write(f'#include "{header_filename}"\n\nnamespace {namespace_name} {{\n\n')
                
                f.write(f"std::vector<GLSLFunction> getFunctionsPart{i+1}() {{\n")
                f.write("    return {\n")
                
                for func_name, func_data in part_functions:
                    # Generate overloads directly in constructor
                    overloads_code = "{"
                    for sig in func_data.overloads:
                        param_list = ', '.join(f'"{p}"' for p in sig.param_types)
                        overloads_code += f'FunctionOverload("{sig.return_type}", {{{param_list}}}), '
                    if overloads_code.endswith(', '):
                        overloads_code = overloads_code[:-2]  # Remove trailing comma
                    overloads_code += "}"
                    
                    f.write(f'        GLSLFunction("{func_name}", "{func_data.file_path}", {overloads_code}),\n')
                
                f.write("    };\n")
                f.write("}\n\n")
                f.write(f"}} // namespace {namespace_name}\n")
            
            generated_files.append(str(part_output_path))
        
        return generated_files

    def generate_cpp_implementation(self, output_filename: str = None) -> str:
        """Generate optimized C++ implementation file (legacy single file version)"""
        if not output_filename:
            output_filename = f"{self.plugin_name.title()}Plugin.cpp"
        
        output_path = self.output_dir / output_filename
        header_filename = f"{self.plugin_name.title()}Plugin.h"
        plugin_info = PluginInfo(name=self.plugin_name, version=self.version, author=self.author)
        namespace_name = f"{self.plugin_name.title()}Plugin"

        with open(output_path, 'w') as f:
            f.write(f'#include "{header_filename}"\n#include <algorithm>\n#include <unordered_map>\n\nnamespace {namespace_name} {{\n\n')
            
            # Plugin info
            f.write(f"""const PluginInfo PLUGIN_INFO("{plugin_info.name}", "{plugin_info.version}", "{plugin_info.author}");

""")
            
            # Step 1: Hardcoded functions array (no emplace_back)
            f.write("const std::vector<GLSLFunction> FUNCTIONS = {\n")
            
            # Pre-calculate all function data with overloads
            sorted_functions = sorted(self.functions.items())
            for func_name, func_data in sorted_functions:
                # Generate overloads directly in constructor
                overloads_code = "{"
                for sig in func_data.overloads:
                    param_list = ', '.join(f'"{p}"' for p in sig.param_types)
                    overloads_code += f'FunctionOverload("{sig.return_type}", {{{param_list}}}), '
                if overloads_code.endswith(', '):
                    overloads_code = overloads_code[:-2]  # Remove trailing comma
                overloads_code += "}"
                
                f.write(f'    GLSLFunction("{func_name}", "{func_data.file_path}", {overloads_code}),\n')
            
            f.write("};\n\n")
            
            # Step 2: Hardcoded hash map for O(1) lookup
            f.write("const std::unordered_map<std::string, const GLSLFunction*> FUNCTION_MAP = {\n")
            
            for i, (func_name, func_data) in enumerate(sorted_functions):
                f.write(f'    {{"{func_name}", &FUNCTIONS[{i}]}},\n')
            
            f.write("};\n\n")
            
            # Optimized utility functions
            f.write("""const GLSLFunction* findFunction(const std::string& name) {
    auto it = FUNCTION_MAP.find(name);
    return (it != FUNCTION_MAP.end()) ? it->second : nullptr;
}

std::vector<std::string> getAllFunctionNames() {
    std::vector<std::string> names;
    for (const auto& func : FUNCTIONS) {
        names.push_back(func.name);
    }
    return names;
}

""")
            f.write(f"}} // namespace {namespace_name}\n")
        
        return str(output_path)
    
    def run(self) -> None:
        """Main processing function"""
        print(f"Starting GLSL plugin generation for: {self.plugin_name}")
        print(f"Input directory: {self.input_root}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all GLSL files
        glsl_files = self.find_glsl_files()
        print(f"Found {len(glsl_files)} GLSL files")
        
        # Process each file
        for file_path in glsl_files:
            self.process_file(file_path)
        
        # Clean up duplicates
        self.remove_duplicate_overloads()
        
        print(f"\nProcessing complete!")
        print(f"Processed {len(self.processed_files)} files")
        print(f"Extracted {len(self.functions)} unique functions")
        
        # Generate output files
        header_path = self.generate_cpp_header()
        
        # Use split implementation for better compilation performance
        impl_files = self.generate_cpp_implementation_split(functions_per_file=50)
        
        plugin_impl_path = self.generate_plugin_implementation()
        cmake_path = self.generate_cmake_file_split(impl_files)
        setup_script_path = self.generate_git_setup_script()
        
        print(f"Generated: {header_path}")
        print(f"Generated {len(impl_files)} implementation files:")
        for file_path in impl_files:
            print(f"  - {file_path}")
        print(f"Generated: {plugin_impl_path}")
        print(f"Generated: {cmake_path}")
        print(f"Generated: {setup_script_path}")
        
        # Print some statistics
        total_overloads = sum(len(func.overloads) for func in self.functions.values())
        print(f"\nStatistics:")
        print(f"- Plugin name: {self.plugin_name}")
        print(f"- Total functions: {len(self.functions)}")
        print(f"- Total overloads: {total_overloads}")
        if len(self.functions) > 0:
            print(f"- Average overloads per function: {total_overloads/len(self.functions):.1f}")
        
        # Show some examples
        print(f"\nSample functions:")
        for i, (name, func) in enumerate(sorted(self.functions.items())):
            if i >= 5:  # Show first 5
                break
            print(f"- {name} ({len(func.overloads)} overloads) in {func.file_path}")

    
    def generate_plugin_implementation(self, output_filename: str = None) -> str:
        """Generate plugin implementation class that inherits from BasePluginImpl"""
        if not output_filename:
            output_filename = f"{self.plugin_name.title()}PluginImpl.cpp"
        
        output_path = self.output_dir / output_filename
        header_filename = f"{self.plugin_name.title()}Plugin.h"
        impl_class_name = f"{self.plugin_name.title()}PluginImpl"
        namespace_name = f"{self.plugin_name.title()}Plugin"
        
        with open(output_path, 'w') as f:
            f.write(f"""#include "{header_filename}"
#include "glsl-plugin-interface/src/BasePluginImpl.h"
#include <set>

namespace {namespace_name} {{

/**
 * @brief Implementation class for {self.plugin_name.title()} plugin
 */
class {impl_class_name} : public BasePluginImpl {{
public:
    {impl_class_name}() : BasePluginImpl(PLUGIN_INFO, FUNCTIONS, FUNCTION_MAP) {{}}
    
    virtual ~{impl_class_name}() = default;
}};

}} // namespace {namespace_name}

// C interface implementation
extern "C" {{
    __attribute__((visibility("default")))
    IPluginInterface* createPlugin() {{
        return new {namespace_name}::{impl_class_name}();
    }}
    
    __attribute__((visibility("default")))
    void destroyPlugin(IPluginInterface* plugin) {{
        delete plugin;
    }}
    
    __attribute__((visibility("default")))
    const char* getPluginInfo() {{
        static std::string info = {namespace_name}::PLUGIN_INFO.name + ":" + 
                                 {namespace_name}::PLUGIN_INFO.version + ":" + 
                                 {namespace_name}::PLUGIN_INFO.author;
        return info.c_str();
    }}
    
    __attribute__((visibility("default")))
    int getPluginABIVersion() {{
        return PLUGIN_ABI_VERSION;
    }}
}}
""")
        
        print(f"Generated plugin implementation: {output_path}")
        return str(output_path)

    def generate_cmake_file_split(self, impl_files: list) -> str:
        """Generate CMakeLists.txt for building the plugin with split implementation files"""
        cmake_path = self.output_dir / "CMakeLists.txt"
        plugin_name = f"{self.plugin_name.title()}Plugin"
        
        # Extract just the filenames from full paths
        source_files = []
        for file_path in impl_files:
            filename = Path(file_path).name
            source_files.append(filename)
        
        # Add the plugin implementation file
        source_files.append(f"{plugin_name}Impl.cpp")
        
        with open(cmake_path, 'w') as f:
            f.write(f"""cmake_minimum_required(VERSION 3.16)
project({plugin_name})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find common interface (adjust path as needed)
if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/glsl-plugin-interface/CMakeLists.txt")
    add_subdirectory(glsl-plugin-interface)
else()
    message(FATAL_ERROR "glsl-plugin-interface not found. Please run: git submodule update --init")
endif()

# Source files for the plugin
set(PLUGIN_SOURCES
""")
            
            # Add each source file
            for source_file in source_files:
                f.write(f"    {source_file}\n")
            
            f.write(f""")

# Create shared library
add_library({plugin_name} SHARED ${{PLUGIN_SOURCES}})

# Link with interface
target_link_libraries({plugin_name} PRIVATE GLSLPluginInterface)

# Set library properties
set_target_properties({plugin_name} PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Compiler flags for shared library
target_compile_options({plugin_name} PRIVATE
    -fvisibility=hidden
    -fPIC
)

# Export symbols for plugin interface
target_compile_definitions({plugin_name} PRIVATE
    PLUGIN_EXPORTS
)

# Install target
install(TARGETS {plugin_name}
    LIBRARY DESTINATION lib/plugins
)

# Copy shader files if they exist
if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/shaders")
    install(DIRECTORY shaders/
        DESTINATION share/{plugin_name.lower()}/shaders
        FILES_MATCHING PATTERN "*.glsl"
    )
endif()
""")
        
        print(f"Generated CMakeLists.txt: {cmake_path}")
        return str(cmake_path)

    def generate_cmake_file(self) -> str:
        """Generate CMakeLists.txt for building the plugin as shared library (legacy single file version)"""
        cmake_path = self.output_dir / "CMakeLists.txt"
        plugin_name = f"{self.plugin_name.title()}Plugin"
        
        with open(cmake_path, 'w') as f:
            f.write(f"""cmake_minimum_required(VERSION 3.16)
project({plugin_name})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find common interface (adjust path as needed)
if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/glsl-plugin-interface/CMakeLists.txt")
    add_subdirectory(glsl-plugin-interface)
else()
    message(FATAL_ERROR "glsl-plugin-interface not found. Please run: git submodule update --init")
endif()

# Create shared library
add_library({plugin_name} SHARED
    {plugin_name}.cpp
    {plugin_name}Impl.cpp
)

# Link with interface
target_link_libraries({plugin_name} PRIVATE GLSLPluginInterface)

# Set library properties
set_target_properties({plugin_name} PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Compiler flags for shared library
target_compile_options({plugin_name} PRIVATE
    -fvisibility=hidden
    -fPIC
)

# Export symbols for plugin interface
target_compile_definitions({plugin_name} PRIVATE
    PLUGIN_EXPORTS
)

# Install target
install(TARGETS {plugin_name}
    LIBRARY DESTINATION lib/plugins
)

# Copy shader files if they exist
if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/shaders")
    install(DIRECTORY shaders/
        DESTINATION share/{plugin_name.lower()}/shaders
        FILES_MATCHING PATTERN "*.glsl"
    )
endif()
""")
        
        print(f"Generated CMakeLists.txt: {cmake_path}")
        return str(cmake_path)

    def generate_git_setup_script(self) -> str:
        """Generate script to set up git submodule"""
        script_path = self.output_dir / "setup_submodule.sh"
        
        with open(script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# Setup script for {self.plugin_name.title()} plugin

echo "Setting up glsl-plugin-interface submodule..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for {self.plugin_name.title()} plugin"
fi

# Add submodule (adjust URL as needed)
if [ ! -d "glsl-plugin-interface" ]; then
    echo "Adding glsl-plugin-interface submodule..."
    # TODO: Replace with actual repository URL
    # git submodule add <actual-repo-url> glsl-plugin-interface
    echo "Please manually add the submodule:"
    echo "git submodule add <repo-url> glsl-plugin-interface"
else
    echo "Submodule already exists, updating..."
    git submodule update --init --recursive
fi

echo "Setup complete!"
echo ""
echo "To build the plugin:"
echo "mkdir build && cd build"
echo "cmake .."
echo "make"
""")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"Generated setup script: {script_path}")
        return str(script_path)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate C++ plugin from GLSL shader library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_glsl_plugin.py myshaders ./shaders ./output
  python3 generate_glsl_plugin.py your_plugin_name /path/to/your_dir_of_.glsl /path/to/output --version 0.0.1 --author "yourname"
        """
    )
    
    parser.add_argument(
        "plugin_name", 
        type=str,
        help="Plugin name (required)"
    )
    
    parser.add_argument(
        "input_dir", 
        type=str,
        help="Input directory containing GLSL files"
    )
    
    parser.add_argument(
        "output_dir",
        type=str, 
        help="Output directory for generated C++ files"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default="0.0.1",
        help="Plugin version (default: 0.0.1)"
    )
    
    parser.add_argument(
        "--author",
        type=str,
        default="Unknown",
        help="Plugin author (default: Unknown)"
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return 1
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        return 1
    
    # Create parser and run
    try:
        parser = GLSLParser(
            plugin_name=args.plugin_name,
            input_root=str(input_path),
            output_dir=args.output_dir,
            version=args.version,
            author=args.author
        )
        parser.run()
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
