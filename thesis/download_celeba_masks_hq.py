import requests

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        
        print(f"save_response_content")

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    print(f"start session")

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    print(f"token = {token}")

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  
    
file_id = "1badu11NqxGf6qM3PTTooQDJvQbejgbTv"
destination = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/CelebAMask-HQ.zip"
download_file_from_google_drive(file_id, destination)