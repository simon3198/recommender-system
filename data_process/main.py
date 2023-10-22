# url 뒤에 "daily" 대신 "-monthly" 붙여야함
from exporter_ import Exporter
from preprocess_ import Preprocesser
from recommend_ import Recommender

if __name__ == "__main__":
    # make directory for key
    # make_dir()

    # get playlist id with channel id
    # get_playlist_id()

    # get video id with playlist id
    # get_video_id()

    # cleanse data
    # video_id_cleansing()

    # get video information with video id
    # get_video_information()

    # get_video_id_by_channelid()

    # get_video_information()

    # get_statistics()

    exporter = Exporter()
    exporter.test()

    preprocessor = Preprocesser()
    # preprocessor.preprocess()
    # preprocessor.save_keywords()
    # preprocessor.create_keywords()
    # preprocessor.create_matrix_csv()

    recommender = Recommender()
    recommender.test()
    recommender.create_full_matrix()
