import subprocess
import os

def cut_video_segment_ffmpeg(input_path, output_path, start_time, end_time):
    """
    Cuts a single video segment using FFmpeg via subprocess.
    Uses -ss (start time) and -to (end time) for precise cutting.
    Uses -c copy for fast, lossless cutting (copies streams without re-encoding).

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path for the output (cut) video file.
        start_time (str): Start time in HH:MM:SS (e.g., "00:00:10").
        end_time (str): End time in HH:MM:SS (e.g., "00:00:15").
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at '{input_path}'")
        return False

    command = [
        'ffmpeg',
        '-ss', start_time,         # Start time
        '-to', end_time,           # End time (exclusive of end_time in some FFmpeg versions, inclusive in others; -t duration is safer if you want a specific length)
        '-i', input_path,         # Input file
        '-c', 'copy',             # Copy audio/video streams without re-encoding (faster, no quality loss)
        output_path               # Output file
    ]

    print(f"Attempting to cut from {start_time} to {end_time} into '{output_path}'...")
    try:
        # Use subprocess.run for cleaner error handling and capturing output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Segment cut successfully: '{output_path}'")
        # Optional: Print FFmpeg's stdout and stderr for debugging if needed
        # print("FFmpeg STDOUT:\n", result.stdout)
        # print("FFmpeg STDERR:\n", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cutting segment: {e}")
        print(f"FFmpeg STDOUT: {e.stdout}")
        print(f"FFmpeg STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: FFmpeg command not found.")
        print("Please ensure FFmpeg is installed and added to your system's PATH.")
        print("You can download it from https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def format_time(seconds):
    """Converts seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# --- Main execution ---
if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your input video file!
    input_video = "data/gol-tec.mp4"

    # Define the segments you want to cut (start_seconds, end_seconds)
    # Using float for seconds for precision, then converting to HH:MM:SS for FFmpeg
    segments = [
        {"start_sec": 0, "end_sec": 6},
        {"start_sec": 38, "end_sec": 40},
        {"start_sec": 44, "end_sec": 46},
        {"start_sec": 50, "end_sec": 52},
    ]

    output_dir = "data/cut_segments" # Directory to save the cut videos
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting video cutting process for '{input_video}'...")

    for i, segment in enumerate(segments):
        start_s = segment["start_sec"]
        end_s = segment["end_sec"]

        # Format times for FFmpeg
        start_hhmmss = format_time(start_s)
        end_hhmmss = format_time(end_s)

        output_file_name = f"segment_{i+1}_{start_s}-{end_s}.mp4"
        output_file_path = os.path.join(output_dir, output_file_name)

        success = cut_video_segment_ffmpeg(input_video, output_file_path, start_hhmmss, end_hhmmss)
        if not success:
            print(f"Failed to cut segment {i+1}. Skipping to next.")
        print("-" * 30)

    print("All cutting operations attempted.")