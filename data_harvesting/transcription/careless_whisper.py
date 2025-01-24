import time

import pytubefix.exceptions
import whisper
import json
from pytubefix import YouTube
from data_harvesting.subtitles.dataset_config import DatasetConfig
import os
from pytubefix.innertube import _default_clients


class CaptainWhisper:
    '''This class downloading youtube video in m4a format and processing transcription of this video using openai-whisper, next deleting this audio.'''

    def __init__(self, num_of_videos_transcript: int, type: str = 'small') -> None:
        '''

        :param num_of_videos_transcript:
        :param type: small -> 2 gb vram o gpu , medium 5 gb vram gpu, tiny/base -> 1 GB vram
        '''
        self.num = num_of_videos_transcript
        self.folder = 'audio'
        self.model = whisper.load_model(type)

    def __load_file(self) -> None:
        """This method loads urls from file"""
        self.videos = []
        with open(DatasetConfig.TO_WHISPE_R, 'r', encoding='utf-8') as f:
            for line in f:
                self.videos.append(json.loads(line))

    def __whisper_activation(self, filepath: str) -> str:
        result = self.model.transcribe(filepath)
        return result['text']

    def __save_transcriptions(self, file_path: str = DatasetConfig.TRANSCRIPTION, json_obj: dict = None,
                              parameter: str = 'a') -> None:
        '''
        This method saves json_obj into jsonlines file
        :param file_path:
        :param json_obj:
        :param parameter:
        :return:
        '''
        with open(file_path, parameter, encoding='utf-8') as f:
            json.dump(json_obj, f, ensure_ascii=False)
            f.write("\n")

    def transcription(self,use_po_token :bool=False) -> None:
        '''
        This method downloads yt video.
        :return:
        '''
        age_restricted=[]
        _default_clients["ANDROID_VR"] = _default_clients["ANDROID_CREATOR"]
        self.__load_file()
        i = 0
        for video in self.videos[:]:
            time.sleep(1)
            if self.num != i:
                try:
                    yt = YouTube(video['url'],use_po_token=use_po_token)
                    audio_stream = yt.streams.get_audio_only()
                    downloaded_file = audio_stream.download(output_path=self.folder)
                    print(f"Pobrano {video['title']}")
                    print(f"Pobrano: {video['title']} do folderu {self.folder}")
                    filepath = os.path.join(self.folder, os.path.basename(downloaded_file))
                    text = self.__whisper_activation(filepath)
                    print(f"Zakończono transkrypcje pliku {os.path.basename(downloaded_file)}")
                    jsonl_obj = {
                        "title": yt.title,
                        "url": yt.watch_url,
                        "recording_length": yt.length,
                        "transcription": text
                    }
                    self.__save_transcriptions(DatasetConfig.TRANSCRIPTION, jsonl_obj, parameter='a')
                    os.remove(filepath)
                    jsonl_obj2 = {
                        "title": yt.title,
                        "url": yt.watch_url
                    }
                    self.__save_transcriptions(DatasetConfig.ALL_SCRAPED_MOVIES, jsonl_obj2,'a')
                    i =i+1
                except pytubefix.exceptions.AgeRestrictedError:
                    print('Film nie został pobrany ze względu na ograniczenia wiekowe')
                    jsonl_obj2 = {
                    "title": yt.title,
                    "url": yt.watch_url
                    }
                    age_restricted.append(jsonl_obj2)
                    self.__save_transcriptions('../data/age_restricted.jsonl',jsonl_obj2,'a')
            else:
                break
        # for video in self.videos:
        #     self.__save_transcriptions(DatasetConfig.TO_WHISPE_R, video, 'a')


if __name__ == '__main__':
    transcript = CaptainWhisper(34)
    transcript.transcription(use_po_token=True)
