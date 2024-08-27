import pickle
import os
from moviepy.editor import VideoFileClip
from absl import app,flags
from glob import glob

## env: prepare_data

flags.DEFINE_string('target',None,help='Current motion to perform action.')
flags.DEFINE_string('source', default=None,help='Absolute path to the experiment result file.')
flags.mark_flag_as_required('source')
flags.mark_flag_as_required('target')
FLAGS = flags.FLAGS

class VideoTrimmer:
    def __init__(self,src_path,target):
        src_path = os.path.join('/home/c1l1mo/projects/VideoAlignment/result',src_path)
        self.src_path = src_path
        self.target = target

        self.source_train = os.path.join(src_path,target + "_train_False.pkl")
        self.source_test = os.path.join(src_path,target + "_test_False.pkl")
        print('self.source_train: ',self.source_train)
        print('self.source_test: ',self.source_test)
        
        self.target_train = self.source_train.replace('_False.pkl','_trimmed.pkl')
        self.target_test = self.source_test.replace('_False.pkl','_trimmed.pkl')

        self.target_video_path = os.path.join(src_path,'trimmed_videos')

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
            end_frame = entry['end_frame']
            clip_file = entry['video_file']

            output_filename= self.trim_and_save_video(video_file, start_frame, end_frame,clip_file)
            result = entry
            result['video_file'] = output_filename
            if not "GT" in self.target:
                result['frame_label'] = result['original_frame_label'][start_frame:end_frame]
            result['seq_len'] = len(result['frame_label'])
            self.trimmed_train_result.append(result)

        for entry in self.test:
            video_file = entry['original_video_file']
            start_frame = entry['start_frame']
            end_frame = entry['end_frame']
            clip_file = entry['video_file']

            output_filename= self.trim_and_save_video(video_file, start_frame, end_frame,clip_file)
            result = entry
            result['video_file'] = output_filename
            if not "GT" in self.target:
                result['frame_label'] = result['original_frame_label'][start_frame:end_frame]
            else:
                result['frame_label'] = result['frame_label']
            result['seq_len'] = len(result['frame_label'])
            self.trimmed_test_result.append(result)
        
        with open(self.target_train, 'wb') as f:
            pickle.dump(self.trimmed_train_result, f)
        with open(self.target_test, 'wb') as f:
            pickle.dump(self.trimmed_test_result, f)

    def trim_and_save_video(self, video_file, start_frame, end_frame,clip_file):
        clip = VideoFileClip(video_file)
        fps = clip.fps

        start_time = start_frame / fps
        end_time = end_frame / fps

        
        try:
            trimmed_clip = clip.subclip(start_time, end_time)
        except:
            raise Exception(
                f"""
                video_file: {video_file} \n 
                start_frame: {start_frame} \n
                end_frame: {end_frame} \n
                start_time: {start_time} \n
                end_time: {end_time} \n
                clip.duration: {clip.duration}
            """
            )

        output_filename = os.path.join(self.target_video_path, f"trimmed_{os.path.basename(clip_file)}")
        print('fps: ' , trimmed_clip.fps)
        trimmed_clip.write_videofile(output_filename)

        clip.close()
        trimmed_clip.close()

        return output_filename

    def run(self):
        import os
        os.makedirs(self.target_video_path, exist_ok=True)
        self.process_videos()
        
def main(_argv):
    trimmer = VideoTrimmer(FLAGS.source,FLAGS.target)
    trimmer.run()

if __name__ == '__main__':
    app.run(main)