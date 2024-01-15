port = 4005
path = require('path')
fs = require('fs')

songsPath = path.join( __dirname, '..', 'Media')
libraryPath = path.join( __dirname, '..', 'library')
bus = require('statebus').serve
  port: port,
  client: (client) ->

    client('songs').to_fetch = (key) ->
      songs = []

      song_dirs = fs.readdirSync( songsPath )

      for song_dir in song_dirs
        if fs.lstatSync( path.join(songsPath, song_dir)   ).isDirectory()
          songs.push(song_dir)

      return {
        key: '/songs',
        songs: songs
      }      


    client('manifest/*').to_fetch = (key) ->
      song_dir = key.split('/')
      song_dir = song_dir[song_dir.length - 1]

      manifest_path = path.join( songsPath, song_dir, 'manifest.json' )
      manifest_json = JSON.parse(fs.readFileSync(manifest_path))

      return {
        key: key,
        manifest: manifest_json
      }      

    client('song_config/*').to_fetch = (key) ->
      song = key.split('/')
      song = song[song.length - 1]

      config_path = path.join( libraryPath, "#{song}.json" )
      config_json = JSON.parse(fs.readFileSync(config_path))

      return {
        key: key,
        config: config_json
      }      


    client('song_config/*').to_save = (obj) ->
      if obj.config        

        song = obj.key.split('/')
        song = song[song.length - 1]

        config_path = path.join( libraryPath, "#{song}.json" )

        fs.writeFileSync(config_path, JSON.stringify(obj.config, null, 2))
        bus.dirty(obj.key)     

      else
        console.error("object incorrect")



    client('reaction/*').to_save = (obj) ->
      vid = obj.reaction.id
      song_dir = obj.song

      manifest_path = path.join( songsPath, song_dir, 'manifest.json' )
      manifest_json = JSON.parse(fs.readFileSync(manifest_path))

      reactions_path = path.join( songsPath, song_dir, 'reactions' )

      reactions = manifest_json["reactions"]
      for reaction in Object.values(reactions)
        if reaction.id == vid
          if reaction['download'] && !obj.reaction.download
            deleteMatchingFilesAndDirsSync(reaction.reactor, reactions_path)

          reaction['explicit'] = obj.reaction.explicit
          reaction['download'] = obj.reaction.download
          break

      fs.writeFileSync(manifest_path, JSON.stringify(manifest_json, null, 2))

      bus.dirty "manifest/#{obj.song}"

    client('reaction_metadata/*').to_fetch = (key) ->
      parts = key.split('/')
      song_dir = parts[parts.length - 2]
      reaction_id = parts[parts.length - 1]

      manifest_path = path.join( songsPath, song_dir, 'manifest.json' )
      manifest_json = JSON.parse(fs.readFileSync(manifest_path))

      reaction = manifest_json.reactions[reaction_id]
      reaction_file_prefix = reaction.file_prefix or reaction.reactor

      metadata_dir = path.join( songsPath, song_dir, 'bounded' )

      alignment_path = path.join(metadata_dir, "#{reaction_file_prefix}-CROSS-EXPANDER.json")

      results = {key}

      if fs.existsSync(alignment_path)
        results.alignment = JSON.parse(fs.readFileSync(alignment_path))

      findMatchingFiles = (directory, pattern) ->

        strip_ext = (f) ->
          parsed = path.parse(f)
          path.join(parsed.dir, parsed.name)

        try
          files = fs.readdirSync(directory)
          matchingFiles = []
          for f in files
            if f?.indexOf(pattern) > -1
              matchingFiles.push( strip_ext(f) )

          return matchingFiles
        catch error
          console.error('Error reading directory:', error)
          return []

      # reactors_pattern = /.*-CROSS-EXPANDER-cropped-.*\.mp4$/
      reactors_pattern = "#{reaction_file_prefix}-CROSS-EXPANDER-cropped-"
      results.reactors = findMatchingFiles(metadata_dir, reactors_pattern)

      isolated_backchannel = "#{reaction_file_prefix}-isolated_backchannel.json"
      results.isolated_backchannel = findMatchingFiles(metadata_dir, isolated_backchannel)

      full_backchannel = "#{reaction_file_prefix}-CROSS-EXPANDER/vocals-post-high-passed.wav"
      results.full_backchannel = findMatchingFiles(metadata_dir, full_backchannel)

      return results  


    client('action/*').to_save = (obj) ->

      reaction_id = obj.reaction_id
      song_dir = obj.song


      manifest_path = path.join( songsPath, song_dir, 'manifest.json' )
      manifest_json = JSON.parse(fs.readFileSync(manifest_path))

      reaction = manifest_json.reactions[reaction_id]
      reaction_file_prefix = reaction.file_prefix or reaction.reactor

      metadata_dir = path.join( songsPath, song_dir, 'bounded' )
      cache_dir = path.join( songsPath, song_dir, '_cache' )

      # reactors_pattern = "#{reaction_file_prefix}-CROSS-EXPANDER-cropped-"
      # isolated_backchannel = "#{reaction_file_prefix}-isolated_backchannel.json"      
      # full_backchannel = "#{reaction_file_prefix}-CROSS-EXPANDER/vocals-post-high-passed.wav"

      if obj.scope == 'alignment' and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-", metadata_dir)
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-", cache_dir)

      else if obj.scope == 'cropped reactors' and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-CROSS-EXPANDER-cropped-", metadata_dir)

      else if obj.scope.startsWith('cropped reactors') and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-CROSS-EXPANDER-cropped-", metadata_dir)
        if obj.scope == 'cropped reactors including coarse'
          deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-CROSS-EXPANDER-coarse_face_position_metadata", metadata_dir)

      else if obj.scope == 'isolated backchannel' and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-isolated_backchannel", metadata_dir)

      else if obj.scope == 'asides' and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-aside-", metadata_dir)


      bus.dirty obj.key



    client('channels').to_fetch = (key) ->
      channels = path.join(songsPath, 'reactor_inventory.json')

      channels_json = JSON.parse(fs.readFileSync(channels))

      return {
        key: key,
        channels: channels_json
      }      


    client('channel/*').to_save = (obj) ->
      channelId = obj.val.channelId
      channel_info = obj.val

      manifest_path = path.join( songsPath, 'reactor_inventory.json' )
      manifest_json = JSON.parse(fs.readFileSync(manifest_path))

      manifest_json[channelId] = channel_info
      
      fs.writeFileSync(manifest_path, JSON.stringify(manifest_json, null, 2))

      bus.dirty "channels"


deleteMatchingFilesAndDirsSync = (name, dir) ->
  files = fs.readdirSync dir, withFileTypes: true
  for file in files
    continue if !file.name?    
    fullPath = path.join dir, file.name
    if fs.lstatSync(fullPath).isDirectory()
      if file.name == name
        fs.rmdirSync fullPath, recursive: true, force: true
        console.log "Deleted directory: #{fullPath}"
      else
        deleteMatchingFilesAndDirsSync name, fullPath
    else
      if path.basename(file.name, path.extname(file.name)) == name
        fs.unlinkSync fullPath
        console.log "Deleted file: #{fullPath}"

deleteFilesAndDirsStartingWithSync = (beginning, dir) ->
  console.log("DELETING #{beginning} from #{dir}, recursively")
  files = fs.readdirSync dir, withFileTypes: true
  for file in files
    continue if !file.name?

    fullPath = path.join dir, file.name
    if fs.lstatSync(fullPath).isDirectory()
      if file.name.startsWith beginning
        fs.rmdirSync fullPath, recursive: true, force: true
        console.log "Deleted directory: #{fullPath}"
      else
        deleteMatchingFilesAndDirsSync beginning, fullPath
    else
      if path.basename(file.name, path.extname(file.name)).startsWith beginning
        fs.unlinkSync fullPath
        console.log "Deleted file: #{fullPath}"




server = "statei://localhost:#{port}"
express = require('express')

bus.http.use('/node_modules', express.static('node_modules'))
bus.http.use('/frontend', express.static('frontend'))
bus.http.use('/media', express.static('../Media'))
bus.http.use('/vendor', express.static('vendor'))
bus.http.use('/library', express.static('../library'))

prefix = ''
bus.http.get '/*', (r,res) => 

  html = """
    <!DOCTYPE html>
    <html>
    <head>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <script type="application/javascript">
      window.statebus_server="#{server}";
    </script>


    <link rel="stylesheet" href="#{prefix}/vendor/spectre.css">
    <link rel="stylesheet" href="#{prefix}/vendor/spectre-exp.css">
    <link rel="stylesheet" href="#{prefix}/vendor/spectre-icons.css">


    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jsoneditor/9.5.3/jsoneditor.min.css">
    <script src="#{prefix}/vendor/jsoneditor.min.js"></script>

    <!--
    <script src="#{prefix}/vendor/jsoneditor-schema.min.js"></script>
    -->

    <script src="#{prefix}/vendor/wavesurfer.js"></script>
    <script src="#{prefix}/vendor/wavesurfer.regions.min.js"></script>

    <script src="#{prefix}/node_modules/statebus/extras/react.js"></script>
    <script src="#{prefix}/node_modules/statebus/extras/sockjs.js"></script>
    <script src="#{prefix}/node_modules/statebus/extras/coffee.js"></script>
    <script src="#{prefix}/node_modules/statebus/statebus.js"></script>
    <script src="#{prefix}/node_modules/statebus/client.js"></script>


    <script history-aware-links root="/" src="#{prefix}/client/earl.coffee"></script>
    <script src="#{prefix}/client/viewport_visibility_sensor.coffee"></script>
    <script src="#{prefix}/client/client.coffee"></script>

    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">


    <style>
      .fa {
        font-family: FontAwesome;  
      }

      video::-webkit-media-controls-fullscreen-button
      {
              display: none !important;
      }


      button {
        border-radius: 8px; 
        border: 1px solid #bbb;
        outline: none;
        padding: 4px 8px;
      }
      button:hover {
        border-color: black;
      }

      .ace-jsoneditor .ace_gutter {
        display: none;
      }

      .ace-jsoneditor .ace_scroller {
        position: static;
      }


      .process-actions button {
        display: inline-block;
        margin-left: 10px;
      }



    </style>


    </head> 
    <body>

    </body>
    </html>
      """

  res.send(html)

