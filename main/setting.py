import os
import glob



def get_videos(dataset_dir):

    vid_paths =sorted(glob.glob(os.path.join(dataset_dir, "*")))
    # extract only video_file_name from file_path
    vid_names = [os.path.splitext(os.path.basename(path))[0] for path in vid_paths]
    vid_len = len(vid_names)

    return vid_paths, vid_names, vid_len
