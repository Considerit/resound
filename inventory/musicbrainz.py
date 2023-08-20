
def download_ren():
    try:
        artist_url = 'http://musicbrainz.org/ws/2/artist/?query=artist:Ren&fmt=json&limit=100'
        response = requests.get(artist_url).json()
        
        if response.get('artists') and len(response['artists']) > 0:
            artist_id = response['artists'][0]['id']
            print('got artist id', artist_id)
            return get_songs_by_artist(artist_id)
        else:
            print("Artist not found")
            return []
    except requests.RequestException as e:
        print(f"Error fetching artist data: {e}")
        return []

def get_songs_by_artist(artist_id):
    songs = []
    try:
        url = f"http://musicbrainz.org/ws/2/recording?artist={artist_id}&fmt=json&limit=100"
        response = requests.get(url).json()
        
        for recording in response.get('recordings', []):
            existing_song = next((song for song in songs if song['name'] == recording['title']), None)
            if not existing_song:
                songs.append({
                    'name': recording['title'],
                    'release_date': recording.get('first-release-date')
                })
        print('HIHI', songs)
        return songs
    except requests.RequestException as e:
        print(f"Error fetching songs data: {e}")
        return []
