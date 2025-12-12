#!/usr/bin/env python3
"""
Audio-Reactive Visualizer System Launcher
A unified CLI for launching visualizers and running post-processing.
"""

import os
import sys
import json
import subprocess
import shlex
import glob
import random
import time
from pathlib import Path
from datetime import datetime

# ==================== CONSTANTS ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
CONFIG_FILE = PROJECT_ROOT / ".launcher_config.json"
TEMP_CENSOR_CONFIG = Path("/tmp/nudity_censor_config.json")
ARCHIVE_VISUALIZER = PROJECT_ROOT / "archive" / "visualizer"
ARCHIVE_POSTPROCESS = PROJECT_ROOT / "archive" / "postprocess"

DEFAULT_CONFIG = {
    "version": "1.0.0",
    "first_run_complete": False,
    "video_library": {
        "base_path": str(PROJECT_ROOT / "clips" / "grok-video-d481b7fd-0998-4b3b-82cc-c9e2a5c1aade"),
        "extension": ".mp4",
        "total_clips": 62
    },
    "nudity_censor": {
        "model_path": str(PROJECT_ROOT / "models" / "640m.onnx"),
        "default_input": "",
        "default_output": ""
    },
    "visualizer_preferences": {
        "startup_delay": 2.0
    },
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "archive_visualizer": str(ARCHIVE_VISUALIZER),
        "archive_postprocess": str(ARCHIVE_POSTPROCESS),
        "venv_python": str(PROJECT_ROOT / ".venv" / "bin" / "python3")
    }
}


# ==================== CONFIG MANAGER ====================
class ConfigManager:
    """Handles loading, saving, and migrating configuration"""

    def __init__(self):
        self.config_path = CONFIG_FILE

    def load(self):
        """Load config from disk, create defaults if missing"""
        if not self.config_path.exists():
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return self._migrate(config)
        except (json.JSONDecodeError, KeyError):
            # Corrupted config - backup and reset
            backup = self.config_path.with_suffix('.json.bak')
            self.config_path.rename(backup)
            print(f"Config corrupted. Backed up to {backup}")
            return DEFAULT_CONFIG.copy()

    def save(self, config):
        """Save config to disk"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def reset_to_defaults(self):
        """Factory reset"""
        if self.config_path.exists():
            self.config_path.unlink()
        return DEFAULT_CONFIG.copy()

    def _migrate(self, old_config):
        """Handle version migrations"""
        version = old_config.get('version', '0.0.0')

        # Ensure all default keys exist
        for key, value in DEFAULT_CONFIG.items():
            if key not in old_config:
                old_config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in old_config[key]:
                        old_config[key][subkey] = subvalue

        old_config['version'] = DEFAULT_CONFIG['version']
        return old_config


# ==================== ARCHIVE BROWSER ====================
class ArchiveBrowser:
    """Manages archive script selection"""

    def __init__(self, archive_path):
        self.archive_path = Path(archive_path)
        self.scripts = self._discover_scripts()

    def _discover_scripts(self):
        """Find all Python scripts in archive"""
        if not self.archive_path.exists():
            return []

        scripts = []
        for i, path in enumerate(sorted(self.archive_path.glob("*.py")), 1):
            # Extract readable name from filename
            name = path.stem
            # Remove leading numbers like "01_"
            display_name = name.lstrip("0123456789_").replace("_", " ").title()

            scripts.append({
                "id": i,
                "filename": path.name,
                "path": str(path),
                "display_name": display_name
            })
        return scripts

    def display_menu(self, page=1, page_size=10):
        """Show paginated archive list"""
        if not self.scripts:
            print("  No archived scripts found.")
            return

        total_pages = (len(self.scripts) + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size
        page_scripts = self.scripts[start:end]

        print(f"\n  Page {page}/{total_pages}")
        print("  " + "─" * 50)
        for script in page_scripts:
            print(f"  [{script['id']:2d}] {script['display_name']}")
            print(f"       ({script['filename']})")
        print("  " + "─" * 50)

        if total_pages > 1:
            print(f"  [N] Next Page  [P] Previous Page")
        print(f"  [R] Random  [B] Back")

        return total_pages

    def get_random(self):
        """Return random script path"""
        if not self.scripts:
            return None
        return random.choice(self.scripts)['path']

    def get_by_id(self, script_id):
        """Get script by menu number"""
        for script in self.scripts:
            if script['id'] == script_id:
                return script['path']
        return None

    def get_count(self):
        """Return number of archived scripts"""
        return len(self.scripts)


# ==================== VALIDATORS ====================
class Validator:
    """Pre-flight checks for launches"""

    @staticmethod
    def validate_visualizer(script1_path, script2_path, config):
        """Check visualizer requirements"""
        errors = []
        warnings = []

        # Check scripts exist
        if not os.path.exists(script1_path):
            errors.append(f"Window 1 script not found: {script1_path}")
        if not os.path.exists(script2_path):
            errors.append(f"Window 2 script not found: {script2_path}")

        # Check video files exist
        base_path = config['video_library']['base_path']
        ext = config['video_library']['extension']
        video_pattern = f"{base_path}*{ext}"
        videos = glob.glob(video_pattern)
        if not videos:
            errors.append(f"No video files found matching: {video_pattern}")
        else:
            expected = config['video_library']['total_clips']
            if len(videos) < expected:
                warnings.append(f"Found {len(videos)} clips (expected {expected})")

        # Check venv exists
        venv_python = config['paths']['venv_python']
        if not os.path.exists(venv_python):
            errors.append(f"Virtual environment not found: {venv_python}")

        return errors, warnings

    @staticmethod
    def validate_censor(input_path, output_path, model_path):
        """Check censor requirements"""
        errors = []
        warnings = []

        # Check input exists
        if not input_path:
            errors.append("No input video specified")
        elif not os.path.exists(input_path):
            errors.append(f"Input video not found: {input_path}")

        # Check output directory is writable
        if output_path:
            output_dir = os.path.dirname(output_path) or "."
            if not os.access(output_dir, os.W_OK):
                errors.append(f"Cannot write to output directory: {output_dir}")

            # Warn if overwriting
            if os.path.exists(output_path):
                warnings.append(f"Output file exists and will be overwritten: {output_path}")
        else:
            errors.append("No output path specified")

        # Check model exists
        if not os.path.exists(model_path):
            errors.append(
                f"AI model not found: {model_path}\n"
                "   Download from: https://github.com/notAI-tech/NudeNet/releases\n"
                "   Get: 640m.onnx and place in models/ folder"
            )

        return errors, warnings


# ==================== LAUNCHER ====================
class Launcher:
    """Handles process spawning"""

    @staticmethod
    def spawn_visualizer(script_path, window_name, venv_python, project_root):
        """Launch visualizer in new Terminal window via osascript"""

        # Simple command - cd and run with python
        cmd = f"cd '{project_root}' && python '{script_path}'"

        applescript = f'''tell application "Terminal"
    do script "{cmd}"
    activate
end tell'''

        try:
            subprocess.run(['osascript', '-e', applescript], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Failed to launch {window_name}: {e}")
            return False

    @staticmethod
    def spawn_censor(input_path, output_path, model_path, venv_python, project_root):
        """Launch nudity censor with temp config"""

        # Write temp config for nudity_censor.py to read
        config = {
            "input_video": input_path,
            "output_video": output_path,
            "model_path": model_path
        }

        with open(TEMP_CENSOR_CONFIG, 'w') as f:
            json.dump(config, f)

        # Run in current terminal (so user can see progress)
        script_path = Path(project_root) / "nudity_censor.py"

        print(f"\n  Starting nudity censor...")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print("  " + "─" * 50)

        try:
            subprocess.run([venv_python, str(script_path)], cwd=project_root)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Censor failed: {e}")
            return False
        finally:
            # Clean up temp config
            if TEMP_CENSOR_CONFIG.exists():
                TEMP_CENSOR_CONFIG.unlink()


# ==================== UI MENUS ====================
class UI:
    """Terminal UI rendering and input handling"""

    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    @staticmethod
    def print_header(title):
        """Print section header"""
        print("\n" + "═" * 55)
        print(f"   {title}")
        print("═" * 55 + "\n")

    @staticmethod
    def print_subheader(title):
        """Print subsection header"""
        print(f"\n{title}")
        print("─" * 55)

    @staticmethod
    def get_input(prompt, default=None):
        """Get user input with optional default"""
        if default:
            display = f"{prompt} [{default}]: "
        else:
            display = f"{prompt}: "

        value = input(display).strip()
        return value if value else default

    @staticmethod
    def confirm(prompt, default=True):
        """Yes/no confirmation"""
        suffix = "[Y/n]" if default else "[y/N]"
        response = input(f"{prompt} {suffix}: ").strip().lower()

        if not response:
            return default
        return response in ('y', 'yes')

    @staticmethod
    def main_menu():
        """Show main menu, return user choice"""
        UI.clear_screen()
        UI.print_header("AUDIO-REACTIVE VISUALIZER SYSTEM")

        print("  [1] Launch Dual-Window Visualizer")
        print("  [2] Run Nudity Censor (Post-Process)")
        print("  [3] Settings / Configuration")
        print("  [4] Exit")
        print()

        while True:
            choice = input("  Choice: ").strip()
            if choice in ('1', '2', '3', '4'):
                return int(choice)
            print("  Invalid choice. Enter 1-4.")

    @staticmethod
    def visualizer_menu(config, archive_browser):
        """Visualizer setup flow - returns (window1_path, window2_path)"""
        UI.clear_screen()
        UI.print_header("DUAL-WINDOW VISUALIZER SETUP")

        project_root = config['paths']['project_root']
        current_w1 = os.path.join(project_root, "visualizer_window1.py")
        current_w2 = os.path.join(project_root, "visualizer_window2.py")

        # Show BlackHole reminder
        print("  Audio Routing Reminder:")
        print("  Make sure BlackHole is configured in Audio MIDI Setup")
        print("  (See Settings menu for setup instructions)")
        print()

        # Window 1 selection
        UI.print_subheader("Window 1 Selection")
        print("  [1] Current (visualizer_window1.py)")
        print(f"  [2] Select from Archive ({archive_browser.get_count()} versions)")
        print("  [3] Random from Archive")
        print()

        window1_path = current_w1
        while True:
            choice = input("  Choice [1]: ").strip() or "1"
            if choice == "1":
                window1_path = current_w1
                break
            elif choice == "2":
                selected = UI.archive_selection_menu(archive_browser)
                if selected:
                    window1_path = selected
                break
            elif choice == "3":
                window1_path = archive_browser.get_random() or current_w1
                print(f"  → Random: {os.path.basename(window1_path)}")
                break
            else:
                print("  Invalid choice.")

        # Window 2 selection
        UI.print_subheader("Window 2 Selection")
        print("  [1] Current (visualizer_window2.py)")
        print(f"  [2] Select from Archive ({archive_browser.get_count()} versions)")
        print("  [3] Random from Archive")
        print("  [R] Random BOTH Windows")
        print()

        window2_path = current_w2
        while True:
            choice = input("  Choice [1]: ").strip() or "1"
            if choice == "1":
                window2_path = current_w2
                break
            elif choice == "2":
                selected = UI.archive_selection_menu(archive_browser)
                if selected:
                    window2_path = selected
                break
            elif choice == "3":
                window2_path = archive_browser.get_random() or current_w2
                print(f"  → Random: {os.path.basename(window2_path)}")
                break
            elif choice.lower() == "r":
                # Random both - ensure different scripts
                scripts = [s['path'] for s in archive_browser.scripts]
                if len(scripts) >= 2:
                    window1_path = random.choice(scripts)
                    remaining = [s for s in scripts if s != window1_path]
                    window2_path = random.choice(remaining)
                elif len(scripts) == 1:
                    window1_path = scripts[0]
                    window2_path = scripts[0]
                print(f"  → Window 1: {os.path.basename(window1_path)}")
                print(f"  → Window 2: {os.path.basename(window2_path)}")
                break
            else:
                print("  Invalid choice.")

        return window1_path, window2_path

    @staticmethod
    def archive_selection_menu(archive_browser):
        """Browse and select from archive"""
        page = 1
        page_size = 10

        while True:
            UI.clear_screen()
            UI.print_header("ARCHIVE BROWSER")

            total_pages = archive_browser.display_menu(page, page_size)
            print()

            choice = input("  Enter number, [N/P/R/B]: ").strip().lower()

            if choice == 'b':
                return None
            elif choice == 'n' and page < total_pages:
                page += 1
            elif choice == 'p' and page > 1:
                page -= 1
            elif choice == 'r':
                return archive_browser.get_random()
            else:
                try:
                    script_id = int(choice)
                    path = archive_browser.get_by_id(script_id)
                    if path:
                        return path
                    print("  Invalid ID.")
                except ValueError:
                    pass

    @staticmethod
    def censor_menu(config):
        """Censor setup flow - returns (input_path, output_path)"""
        UI.clear_screen()
        UI.print_header("NUDITY CENSOR - POST-PROCESSING")

        model_path = config['nudity_censor']['model_path']

        # Model check
        if os.path.exists(model_path):
            print(f"  ✓ AI Model found: {os.path.basename(model_path)}")
        else:
            print(f"  ✗ AI Model NOT found!")
            print(f"    Expected: {model_path}")
            print(f"    Download from: https://github.com/notAI-tech/NudeNet/releases")
            print()
            if not UI.confirm("  Continue anyway?", default=False):
                return None, None

        print()

        # Input file
        default_input = config['nudity_censor'].get('default_input', '')
        UI.print_subheader("Input Video")
        if default_input:
            print(f"  Last used: {default_input}")

        input_path = UI.get_input("  Path to input video", default_input)

        if not input_path:
            print("  No input specified.")
            input("  Press Enter to go back...")
            return None, None

        # Expand ~ in path
        input_path = os.path.expanduser(input_path)

        # Output file
        default_output = config['nudity_censor'].get('default_output', '')
        if not default_output and input_path:
            # Generate default output name
            base, ext = os.path.splitext(input_path)
            default_output = f"{base}_censored{ext}"

        UI.print_subheader("Output Video")
        output_path = UI.get_input("  Path for output video", default_output)

        if not output_path:
            print("  No output specified.")
            input("  Press Enter to go back...")
            return None, None

        output_path = os.path.expanduser(output_path)

        return input_path, output_path

    @staticmethod
    def settings_menu(config):
        """Settings configuration"""
        while True:
            UI.clear_screen()
            UI.print_header("CONFIGURATION")

            print("  Video Library:")
            print(f"    Base Path: {config['video_library']['base_path']}")
            print(f"    Extension: {config['video_library']['extension']}")
            print(f"    Clips: {config['video_library']['total_clips']}")
            print()
            print("  Nudity Censor:")
            print(f"    Model: {config['nudity_censor']['model_path']}")
            print(f"    Default Input: {config['nudity_censor'].get('default_input', '(none)')}")
            print(f"    Default Output: {config['nudity_censor'].get('default_output', '(none)')}")
            print()
            print("  " + "─" * 50)
            print("  [1] Change Video Library Path")
            print("  [2] Change Default Censor Input")
            print("  [3] Change Default Censor Output")
            print("  [4] Show BlackHole Setup Instructions")
            print("  [5] Reset to Defaults")
            print("  [B] Back to Main Menu")
            print()

            choice = input("  Choice: ").strip().lower()

            if choice == 'b':
                return config
            elif choice == '1':
                new_path = UI.get_input("  New base path", config['video_library']['base_path'])
                config['video_library']['base_path'] = new_path
            elif choice == '2':
                new_path = UI.get_input("  Default input path", config['nudity_censor'].get('default_input', ''))
                config['nudity_censor']['default_input'] = new_path
            elif choice == '3':
                new_path = UI.get_input("  Default output path", config['nudity_censor'].get('default_output', ''))
                config['nudity_censor']['default_output'] = new_path
            elif choice == '4':
                UI.show_blackhole_instructions()
            elif choice == '5':
                if UI.confirm("  Reset all settings to defaults?", default=False):
                    config = DEFAULT_CONFIG.copy()
                    print("  Settings reset.")
                    input("  Press Enter to continue...")

        return config

    @staticmethod
    def show_blackhole_instructions():
        """Display BlackHole audio setup instructions"""
        UI.clear_screen()
        UI.print_header("BLACKHOLE AUDIO SETUP")

        print("""
  BlackHole is a virtual audio driver that lets you route
  system audio to the visualizer.

  INSTALLATION:
  ─────────────────────────────────────────────────────────
  brew install blackhole-2ch

  CONFIGURATION:
  ─────────────────────────────────────────────────────────
  1. Open "Audio MIDI Setup" (search in Spotlight)

  2. Click [+] at bottom left → "Create Multi-Output Device"

  3. In the new device, check:
     ✓ Your Speakers/Headphones
     ✓ BlackHole 2ch

  4. Right-click the Multi-Output Device →
     "Use This Device for Sound Output"

  5. Now audio plays through speakers AND routes to BlackHole

  The visualizer will auto-detect BlackHole as input.
  If it doesn't, you'll be prompted to select it manually.

  ─────────────────────────────────────────────────────────
        """)

        input("  Press Enter to go back...")

    @staticmethod
    def first_run_wizard(config):
        """Initial setup wizard"""
        UI.clear_screen()
        UI.print_header("FIRST RUN - SETUP WIZARD")

        print("  Welcome! Let's configure your system.\n")

        # Step 1: Video Library
        UI.print_subheader("Step 1/2: Video Library Location")

        base_path = config['video_library']['base_path']
        ext = config['video_library']['extension']
        videos = glob.glob(f"{base_path}*{ext}")

        if videos:
            print(f"  Found: {len(videos)} video clips")
            print(f"  Path: {base_path}")
            if not UI.confirm("\n  Use this path?", default=True):
                new_path = UI.get_input("  Enter new base path")
                if new_path:
                    config['video_library']['base_path'] = new_path
        else:
            print(f"  No videos found at default path.")
            new_path = UI.get_input("  Enter path to video files (without extension)")
            if new_path:
                config['video_library']['base_path'] = new_path

        # Step 2: BlackHole
        UI.print_subheader("Step 2/2: Audio Routing (BlackHole)")

        print("  For audio-reactive visuals, you need BlackHole installed.")

        if not UI.confirm("\n  Have you configured BlackHole?", default=True):
            UI.show_blackhole_instructions()

        print("\n  Setup complete!")
        input("  Press Enter to continue to main menu...")

        config['first_run_complete'] = True
        return config


# ==================== MAIN ====================
def main():
    """Entry point"""

    config_mgr = ConfigManager()
    config = config_mgr.load()

    # First run wizard
    if not config.get('first_run_complete'):
        config = UI.first_run_wizard(config)
        config_mgr.save(config)

    archive = ArchiveBrowser(config['paths']['archive_visualizer'])

    while True:
        try:
            choice = UI.main_menu()

            if choice == 1:
                # Visualizer mode
                window1, window2 = UI.visualizer_menu(config, archive)

                errors, warnings = Validator.validate_visualizer(window1, window2, config)

                if warnings:
                    print("\n  Warnings:")
                    for w in warnings:
                        print(f"    ⚠ {w}")

                if errors:
                    print("\n  Errors:")
                    for e in errors:
                        print(f"    ✗ {e}")
                    input("\n  Press Enter to go back...")
                    continue

                # Confirm launch
                print(f"\n  Window 1: {os.path.basename(window1)}")
                print(f"  Window 2: {os.path.basename(window2)}")

                if UI.confirm("\n  Launch now?", default=True):
                    venv = config['paths']['venv_python']
                    root = config['paths']['project_root']
                    delay = config['visualizer_preferences'].get('startup_delay', 2.0)

                    print("\n  Launching Window 1...")
                    Launcher.spawn_visualizer(window1, "Visualizer 1", venv, root)

                    print(f"  Waiting {delay}s for initialization...")
                    time.sleep(delay)

                    print("  Launching Window 2...")
                    Launcher.spawn_visualizer(window2, "Visualizer 2", venv, root)

                    print("\n  ✓ Both windows launched!")
                    print("    Press 'q' in each OpenCV window to quit")
                    input("\n  Press Enter to return to menu...")

            elif choice == 2:
                # Censor mode
                input_path, output_path = UI.censor_menu(config)

                if not input_path or not output_path:
                    continue

                model_path = config['nudity_censor']['model_path']
                errors, warnings = Validator.validate_censor(input_path, output_path, model_path)

                if warnings:
                    print("\n  Warnings:")
                    for w in warnings:
                        print(f"    ⚠ {w}")
                    if not UI.confirm("\n  Continue?", default=True):
                        continue

                if errors:
                    print("\n  Errors:")
                    for e in errors:
                        print(f"    ✗ {e}")
                    input("\n  Press Enter to go back...")
                    continue

                # Save as new defaults
                config['nudity_censor']['default_input'] = input_path
                config['nudity_censor']['default_output'] = output_path
                config_mgr.save(config)

                # Run censor
                venv = config['paths']['venv_python']
                root = config['paths']['project_root']
                Launcher.spawn_censor(input_path, output_path, model_path, venv, root)

                input("\n  Press Enter to return to menu...")

            elif choice == 3:
                # Settings
                config = UI.settings_menu(config)
                config_mgr.save(config)

            elif choice == 4:
                # Exit
                print("\n  Goodbye!")
                break

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Returning to menu...")
            continue


if __name__ == "__main__":
    main()
