from google_images_search import GoogleImagesSearch
import dotenv
import os

dotenv.load_dotenv()
developer_key = os.getenv("DEVELOPER_KEY")
custom_search_cx = os.getenv("CUSTOM_SEARCH_CX")


def my_progressbar(url, progress):
    print(url + ' ' + str(progress) + '%')


def download_images(query, limit):
    gis = GoogleImagesSearch(developer_key, custom_search_cx,
                             progressbar_fn=my_progressbar)
    # define search params:
    _search_params = {
        'q': query,
        'num': limit,
        'fileType': 'jpg|gif|png',
        'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
        'safe': 'active',
        'imgSize': 'xxlarge',
    }

    gis.search(search_params=_search_params, path_to_dir='image', width=500, height=500)


query = "woman"
limit = 5
download_images(query, limit)
