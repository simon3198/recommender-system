# from data_process import exporter_, preprocess_, recommend_, setting

from data_process import Exporter, Preprocesser, Recommender

if __name__ == "__main__":
    exporter = Exporter()

    # make directory for key
    # exporter.make_dir()

    # get playlist id with channel id
    # exporter.get_playlist_id()

    # get video id with playlist id
    # exporter.get_video_id()

    # cleanse data
    # exporter.video_id_cleansing()

    # get video information with video id
    # exporter.get_video_information()
    # exporter.get_video_id_by_channelid()
    # exporter.get_video_information()
    # exporter.get_statistics()

    # preprocess the raw data
    # preprocessor = Preprocesser()
    # preprocessor.preprocess()
    # preprocessor.save_keywords()
    # preprocessor.create_keywords()
    # preprocessor.create_matrix_csv()

    # recommender
    recommender = Recommender()
    # recommender.test()
    # recommender.create_full_matrix_svd()
    # recommender.create_full_matrix_coll_keyword_based()
    # recommender.create_full_matrix_coll_inf_based()
    # recommender.create_full_matrix_als()
    # recommender.create_full_matrix_mlp_mf()

    # recommender.create_full_matrix_ncf()

    # model_type = "svd"
    model_list = ["svd", "coll_keyword", "coll_inf", "als", "mlp_mf", "ncf"]

    # for model_type in model_list:
    #     recommender.get_metric_value(model_type)
    #     print(model_type)

    # 모델 하나당 총 60개의 matrix이 생성된다.
    # metrix(view,likes,comment) 별로 할 것이냐
    # 아니면 카테고리별로 최고의 모델을 뽑을 것이냐
    # 아니면 종합해서 뽑을것이냐 (이건 이미 함)

    # 둘 다 해보고 좋은걸로 하자

    # recommender.calculate_eval_by_category()
    recommender.calculate_eval_by_metrix()
