import os
import subprocess
import sys


def install_ffmpeg():
    print("Starting Ffmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])

    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "setuptools"])

    try:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()

        user_bin_dir = os.path.expanduser("~/.local/bin")
        os.makedirs(user_bin_dir, exist_ok=True)
        target_path = os.path.join(user_bin_dir, "ffmpeg")

        subprocess.check_call(["cp", ffmpeg_path, target_path])

        subprocess.check_call(["chmod", "+x", target_path])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    try:
        ffmpeg_bin = os.path.expanduser("~/.local/bin/ffmpeg")
        cmd = [ffmpeg_bin, "-version"] if os.path.exists(ffmpeg_bin) else ["ffmpeg", "-version"]
        result = subprocess.run(cmd,
                                capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False
    
if __name__ == "__main__":
    install_ffmpeg()