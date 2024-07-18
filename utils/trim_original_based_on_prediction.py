import pickle
import os
from moviepy.editor import VideoFileClip

class VideoTrimmer:
    def __init__(self):
        self.source_train = '/home/c1l1mo/projects/VideoAlignment/result/one_jump/output_new_train_label.pkl'
        self.source_test = '/home/c1l1mo/projects/VideoAlignment/result/one_jump/output_new_test_label.pkl'
        self.target_train = '/home/c1l1mo/projects/VideoAlignment/result/one_jump/output_new_train_trimmed.pkl'
        self.target_test = '/home/c1l1mo/projects/VideoAlignment/result/one_jump/output_new_test_trimmed.pkl'
        self.target_video_path = '/home/c1l1mo/projects/VideoAlignment/result/one_jump/trimmed_videos/'

    def process_videos(self):
        with open(self.source_train, 'rb') as f:
            self.train = pickle.load(f)
        with open(self.source_test, 'rb') as f:
            self.test = pickle.load(f)

        self.trimmed_train_result = []
        self.trimmed_test_result = []
        for entry in self.train:
            video_file = entry['original_video_file']
            start_frame = entry['start_frame']
            end_frame = start_frame + len(entry['subtraction'])

            output_filename= self.trim_and_save_video(video_file, start_frame, end_frame)
            result = entry
            result['video_file'] = output_filename
            result['frame_label'] = result['original_frame_label'][start_frame:end_frame]
            result['seq_len'] = len(result['frame_label'])
            self.trimmed_train_result.append(result)

        for entry in self.test:
            video_file = entry['original_video_file']
            start_frame = entry['start_frame']
            end_frame = start_frame + len(entry['subtraction'])

            output_filename= self.trim_and_save_video(video_file, start_frame, end_frame)
            result = entry
            result['video_file'] = output_filename
            result['frame_label'] = result['original_frame_label'][start_frame:end_frame]
            result['seq_len'] = len(result['frame_label'])
            self.trimmed_test_result.append(result)
        
        with open(self.target_train, 'wb') as f:
            pickle.dump(self.trimmed_train_result, f)
        with open(self.target_test, 'wb') as f:
            pickle.dump(self.trimmed_test_result, f)

    def trim_and_save_video(self, video_file, start_frame, end_frame):
        clip = VideoFileClip(video_file)
        fps = clip.fps

        start_time = start_frame / fps
        end_time = end_frame / fps

        print(start_frame, end_frame, start_time, end_time,clip.duration)

        trimmed_clip = clip.subclip(start_time, end_time)

        output_filename = os.path.join(self.target_video_path, f"trimmed_{os.path.basename(video_file)}")
        trimmed_clip.write_videofile(output_filename)

        clip.close()
        trimmed_clip.close()

        return output_filename

    def run(self):
        import os
        os.makedirs(self.target_video_path, exist_ok=True)
        self.process_videos()
if __name__ == '__main__':
    trimmer = VideoTrimmer()
    trimmer.run()