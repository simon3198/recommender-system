# from data_process import exporter_, preprocess_, recommend_, setting

from data_process import Exporter, Preprocesser, Recommender, setting

if __name__ == "__main__":
    exporter = Exporter()

    # make directory for key
    exporter.make_dir()

    # get playlist id with channel id
    exporter.get_playlist_id()

    # get video id with playlist id
    exporter.get_video_id()

    # cleanse data
    exporter.video_id_cleansing()

    # get video information with video id
    exporter.get_video_information()
    exporter.get_video_id_by_channelid()
    exporter.get_video_information()
    exporter.get_statistics()

    # preprocess the raw data
    preprocessor = Preprocesser()
    preprocessor.preprocess()
    preprocessor.save_keywords()
    preprocessor.create_keywords()
    preprocessor.create_matrix_csv()

    # recommender
    recommender = Recommender()
    recommender.test()
    recommender.create_full_matrix()
