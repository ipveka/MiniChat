"""
Build script for creating MiniChat desktop executables.
Uses PyInstaller to create standalone applications for Windows, macOS, and Linux.
"""

import subprocess
import sys
import os
import shutil
import platform

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MINILM_ROOT = os.path.join(PROJECT_ROOT, 'miniLM')
DIST_DIR = os.path.join(PROJECT_ROOT, 'dist')
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build')


def install_pyinstaller():
    """Install PyInstaller if not present."""
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])


def get_platform_name():
    """Get a friendly platform name."""
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    return system


def build_executable():
    """Build the executable using PyInstaller."""
    install_pyinstaller()
    
    desktop_script = os.path.join(MINILM_ROOT, 'src', 'desktop.py')
    
    # PyInstaller options
    options = [
        desktop_script,
        '--name', 'MiniChat',
        '--onedir',  # Create a directory with executable
        '--windowed',  # No console window
        '--noconfirm',  # Overwrite without asking
        '--clean',  # Clean cache before building
        # Add data files
        '--add-data', f'{os.path.join(MINILM_ROOT, "src")}:miniLM/src',
        '--add-data', f'{os.path.join(MINILM_ROOT, "config")}:miniLM/config',
        # Hidden imports that PyInstaller might miss
        '--hidden-import', 'streamlit',
        '--hidden-import', 'chromadb',
        '--hidden-import', 'sentence_transformers',
        '--hidden-import', 'ollama',
        '--hidden-import', 'langchain',
        '--hidden-import', 'langchain_community',
        '--hidden-import', 'pypdf',
        '--hidden-import', 'docx',
        '--hidden-import', 'webview',
        # Output directory
        '--distpath', DIST_DIR,
        '--workpath', BUILD_DIR,
    ]
    
    # Platform-specific options
    system = platform.system()
    if system == 'Darwin':
        # macOS: Create .app bundle
        options.extend([
            '--osx-bundle-identifier', 'com.minichat.app',
        ])
    elif system == 'Windows':
        # Windows: Add icon if available
        icon_path = os.path.join(MINILM_ROOT, 'installer', 'icon.ico')
        if os.path.exists(icon_path):
            options.extend(['--icon', icon_path])
    
    print(f"Building MiniChat for {get_platform_name()}...")
    print(f"Output directory: {DIST_DIR}")
    
    # Run PyInstaller
    subprocess.check_call([sys.executable, '-m', 'PyInstaller'] + options)
    
    # Create data directories in the output
    output_dir = os.path.join(DIST_DIR, 'MiniChat')
    if system == 'Darwin':
        output_dir = os.path.join(DIST_DIR, 'MiniChat.app', 'Contents', 'MacOS')
    
    for subdir in ['data', 'logs']:
        dir_path = os.path.join(output_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        # Create .gitkeep to preserve directory
        with open(os.path.join(dir_path, '.gitkeep'), 'w') as f:
            pass
    
    print(f"\nBuild complete! Executable is in: {output_dir}")
    print("\nTo run:")
    if system == 'Windows':
        print(f"  {os.path.join(output_dir, 'MiniChat.exe')}")
    elif system == 'Darwin':
        print(f"  open {os.path.join(DIST_DIR, 'MiniChat.app')}")
    else:
        print(f"  {os.path.join(output_dir, 'MiniChat')}")


def clean():
    """Clean build artifacts."""
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(dir_path):
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)
    
    # Remove .spec file
    spec_file = os.path.join(PROJECT_ROOT, 'MiniChat.spec')
    if os.path.exists(spec_file):
        os.remove(spec_file)
    
    print("Clean complete.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        clean()
    else:
        build_executable()


if __name__ == '__main__':
    main()
