port = 4005
path = require('path')
fs = require('fs')

songsPath = path.join( __dirname, '..', 'Media')

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
  files = fs.readdirSync(dir, { withFileTypes: true })

  for file in files
      fullPath = path.join(dir, file)

      if fs.lstatSync( fullPath   ).isDirectory()
          if file == name
              fs.rmdirSync(fullPath, { recursive: true })
              console.log("Deleted directory: #{fullPath}")
          else
              deleteMatchingFilesAndDirsSync(name, fullPath)
      else
          if (path.basename(file, path.extname(file)) == name)
              fs.unlinkSync(fullPath);
              console.log("Deleted file: #{fullPath}") 

deleteFilesAndDirsStartingWithSync = (beginning, dir) ->
  files = fs.readdirSync(dir, { withFileTypes: true })
  for file in files
      fullPath = path.join(dir, file)
      if fs.lstatSync( fullPath   ).isDirectory()
          if file.startsWith(beginning)
              fs.rmdirSync(fullPath, { recursive: true })
              console.log("Deleted directory: #{fullPath}")
          else
              deleteMatchingFilesAndDirsSync(beginning, fullPath)
      else
          if path.basename(file, path.extname(file)).startsWith(beginning)
              fs.unlinkSync(fullPath)
              console.log("Deleted file: #{fullPath}") 




server = "statei://localhost:#{port}"
express = require('express')

bus.http.use('/node_modules', express.static('node_modules'))
bus.http.use('/frontend', express.static('frontend'))
bus.http.use('/media', express.static('../Media'))
bus.http.use('/vendor', express.static('vendor'))

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

    <style>
      .fa {
        font-family: FontAwesome;  
      }

    </style>


    </head> 
    <body>

    </body>
    </html>
      """

  res.send(html)

