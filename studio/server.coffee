
require('dotenv').config()
wavFileInfo = require('wav-file-info')
port = 4005
path = require('path')
fs = require('fs')
sqlite3 = require('sqlite3').verbose()
{execSync} = require 'child_process'

songsPath = path.join( __dirname, '..', 'Media')
libraryPath = path.join( __dirname, '..', 'library')
resoundioDBPath = process.env.RESOUNDIO_DB_PATH
resoundioDevDBPath = process.env.RESOUNDIO_DEV_DB_PATH



get_manifest_path = (song) ->

  manifest_path = path.join( songsPath, song, 'manifest.json' )
  return manifest_path

read_manifest = (song) -> 
  manifest_path = get_manifest_path(song)
  manifest_json = JSON.parse(fs.readFileSync(manifest_path))
  return manifest_json

write_manifest = (song, updated_manifest) ->
  manifest_path = get_manifest_path(song)
  fs.writeFileSync(manifest_path, JSON.stringify(updated_manifest, null, 2))
  bus.dirty "manifest/#{song}"
  
get_layout_path = (song) -> 
  temp_directory = path.join( songsPath, song, 'bounded')
  layout_path = null
  return null unless fs.existsSync(temp_directory)

  files = fs.readdirSync temp_directory
  for file in files
      if file.startsWith('layout-current') and path.extname(file) == '.json'
          layout_path = path.join temp_directory, file
          break
  return layout_path

read_layout = (song) ->
  layout_path = get_layout_path(song)
  if layout_path 
    layout_json = JSON.parse(fs.readFileSync(layout_path))
  else
    layout_json = {}
  return layout_json

write_layout = (song, obj) ->
  layout_path = get_layout_path(song)
  if layout_path
    fs.writeFileSync(layout_path, JSON.stringify(obj.layout, null, 2))
    bus.dirty(obj.key)

get_song_config_path = (song) ->
  config_path = path.join( libraryPath, "#{song}.json" )
  return config_path

read_song_config = (song) ->
  config_path = path.join( libraryPath, "#{song}.json" )
  if fs.existsSync(config_path)
    config_json = JSON.parse(fs.readFileSync(config_path))
  else
    config_json = {}
  return config_json

write_song_config = (song, obj) ->
  config_path = get_song_config_path(song)
  fs.writeFileSync(config_path, JSON.stringify(obj.config, null, 2))
  bus.dirty(obj.key)




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
      song = song_dir[song_dir.length - 1]

      manifest_json = read_manifest(song)

      return {
        key: key,
        manifest: manifest_json
      }      

    client('song_config/*').to_fetch = (key) ->
      song = key.split('/')
      song = song[song.length - 1]

      config_json = read_song_config(song)
      layout_json = read_layout(song)

      return {
        key: key,
        config: config_json,
        layout: layout_json
      }


    client('song_config/*').to_save = (obj) ->
      if obj.config        

        song = obj.key.split('/')
        song = song[song.length - 1]

        write_song_config(song, obj)        

        if obj.layout
          write_layout(song, obj)

      else
        console.error("object incorrect")


      bus.dirty obj.key



    client('reaction/*').to_save = (obj) ->
      vid = obj.reaction.id
      song = obj.song

      manifest_json = read_manifest(song) 

      reactions_path = path.join( songsPath, song, 'reactions' )

      reactions = manifest_json["reactions"]
      for reaction in Object.values(reactions)
        if reaction.id == vid
          if reaction['download'] && !obj.reaction.download
            deleteMatchingFilesAndDirsSync(reaction.reactor, reactions_path)

          reaction['explicit'] = obj.reaction.explicit
          reaction['download'] = obj.reaction.download
          if obj.reaction.marked?
            reaction['marked'] = obj.reaction.marked
          break

      write_manifest(song, manifest_json)




    getWavDurationSync = (filePath) ->
      if !fs.existsSync filePath
        console.error("File does not exist: #{filePath}")
        return 5000

      try
        command = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"#{filePath}\""
        output = execSync command, encoding: 'utf-8'
        durationInSeconds = parseFloat output
        return durationInSeconds
      catch error
        throw new Error 'Failed to get audio duration'

    get_alignment_data = (reaction, song, reaction_file_prefix) ->      
      metadata_dir = path.join( songsPath, song, 'bounded' )
      alignment_path = path.join(metadata_dir, "#{reaction_file_prefix}-CROSS-EXPANDER.json")

      if fs.existsSync(alignment_path)
        alignment_data = JSON.parse(fs.readFileSync(alignment_path))
      else
        return null

      sample_rate = 44100

      if reaction.duration
        [minutes, seconds] = reaction.duration.split(':')
        duration = parseInt(minutes) * 60 + parseInt(seconds)
      else
        reaction_wav = path.join(songsPath, song, 'reactions', "#{reaction_file_prefix}.wav")
        duration = getWavDurationSync(reaction_wav)
        console.log("CALCULSTED DURATION OF #{reaction_wav} as #{duration}")

      

      unaligned_segments = []
      keypoints = []
      
      last_reaction_end = 0

      for segment, idx in alignment_data.best_path

        reaction_start = segment[0] / sample_rate
        reaction_end = segment[1] / sample_rate
        base_start = segment[2] / sample_rate
        base_end = segment[3] / sample_rate


        if last_reaction_end < reaction_start - 1
          unaligned_segments.push 
            type: 'speaking'
            length: reaction_start - last_reaction_end
            start: last_reaction_end
            end: reaction_start
            base_start: base_start
            base_end: base_end

        unaligned_segments.push 
          type: 'backchannel'
          length: reaction_end - reaction_start
          start: reaction_start
          end: reaction_end
          base_start: base_start
          base_end: base_end

        pause_duration = reaction_start - last_reaction_end
        last_reaction_end = reaction_end

      unaligned_segments.push 
        type: 'speaking'
        length: duration - last_reaction_end
        start: last_reaction_end
        end: duration
        base_start: base_end
        base_end: base_end


      for seg in unaligned_segments
        if seg.type == 'speaking'
          if seg.start == 0
            keypoints.push( [seg.end, 0] )
          else
            keypoints.push( [seg.start, seg.base_start] )


      return {
        alignment: alignment_data,
        unaligned_segments: unaligned_segments,
        keypoints: keypoints,
        alignment_score: alignment_data.best_path_score[2]
      }

      

    client('reaction_metadata/*').to_fetch = (key) ->
      parts = key.split('/')
      song = parts[parts.length - 2]
      reaction_id = parts[parts.length - 1]

      manifest_json = read_manifest(song)

      reaction = manifest_json.reactions[reaction_id]

      reaction_file_prefix = reaction.file_prefix or reaction.reactor

      metadata_dir = path.join( songsPath, song, 'bounded' )

      alignment_data = get_alignment_data(reaction, song, reaction_file_prefix)

      results = {key}

      if alignment_data
        for k,v of alignment_data
          results[k] = v

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
      song = obj.song

      manifest_json = read_manifest(song)

      reaction = manifest_json.reactions[reaction_id]
      reaction_file_prefix = reaction.file_prefix or reaction.reactor

      metadata_dir = path.join( songsPath, song, 'bounded' )
      cache_dir = path.join( songsPath, song, '_cache' )

      # reactors_pattern = "#{reaction_file_prefix}-CROSS-EXPANDER-cropped-"
      # isolated_backchannel = "#{reaction_file_prefix}-isolated_backchannel.json"      
      # full_backchannel = "#{reaction_file_prefix}-CROSS-EXPANDER/vocals-post-high-passed.wav"

      if obj.scope == 'alignment' and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-", metadata_dir)
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-", cache_dir)

      else if obj.scope.startsWith('cropped reactors') and obj.action == 'delete'
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-CROSS-EXPANDER-cropped-", metadata_dir)
        deleteFilesAndDirsStartingWithSync("#{reaction_file_prefix}-aside-", metadata_dir)

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


    client('sync_with_resoundio/*').to_save = (obj) ->
      song = obj.song

      song_config = read_song_config(song)
      manifest = read_manifest(song)

      song_vid = manifest['main_song']['id']

      dbs = [resoundioDevDBPath, resoundioDBPath]

      for db_path in dbs

        db = new sqlite3.Database(db_path)

        db.get "SELECT * from User WHERE email='#{process.env.DEFAULT_USER}'", (err, row) =>
          console.log(row)
          default_user_id = row["user_id"]


          ############################
          # Synchronize the base video

          status = 1 # pinned

          song_vid_values = {
            $vid: song_vid, 
            $channel: manifest['main_song']['artist'], 
            $channel_id: manifest['main_song']['channel_id'], 
            $title: manifest['main_song']['title'], 
            $description: manifest['main_song']['description'], 
            $views: manifest['main_song']['views'], 
            $duration: manifest['main_song']['duration']
          }
          song_values = {
            $vid: song_vid, 
            $status: status,  # make sure to put all 
            $song_key: song,
            $added_by: default_user_id 
          }

          db.get "SELECT * FROM Song WHERE vid='#{song_vid}'", (err, row) =>
            if !row?
              console.log("INSERTING NEW VIDEO #{song}")
              db.run \
                """INSERT INTO Video (vid,  channel,  channel_id,  title,  description,  views,  duration) VALUES 
                                    ($vid, $channel, $channel_id, $title, $description, $views, $duration)""", 
                song_vid_values


              console.log("INSERTING NEW SONG #{song}")
              db.run \
                "INSERT INTO Song (vid, song_key, status, added_by) VALUES ($vid, $song_key, $status, $added_by)", 
                song_values
            
            else
              console.log("UPDATING VIDEO #{song}")

              db.run \
                """UPDATE Video SET channel=$channel, channel_id=$channel_id, title=$title, description=$description, views=$views, duration=$duration 
                                    WHERE vid=$vid""", song_vid_values

              console.log("UPDATING SONG #{song}")

              song_values = {  # annoying, node-sqlite3 errors if there are unused values
                $vid: song_values.$vid, 
                $status: song_values.$status
              }
              db.run "UPDATE Song SET status=$status WHERE vid=$vid", song_values




          reactions = manifest["reactions"]
          included_reactions = {}
          for reaction in Object.values(reactions)
            if reaction['download']
              included_reactions[reaction.id] = reaction

          ##################################################
          # Delete any reactions that are no longer included

          db.each "SELECT * FROM Reaction WHERE song_key='#{song}'", (err, row) =>
            if row.vid not of included_reactions
              db.run "DELETE FROM Reaction WHERE vid='#{row.vid}'"
              db.run "DELETE FROM Video WHERE vid='#{row.vid}'"

          ##################################################
          # Insert/update all included reactions 
          
          for reaction_vid, reaction of included_reactions
            do(reaction_vid, reaction) =>
              db.get "SELECT * FROM Reaction WHERE vid='#{reaction_vid}'", (err, row) =>
                # console.log("PROCESSING", row)

                reaction_file_prefix = reaction.file_prefix or reaction.reactor
                alignment = get_alignment_data(reaction, song, reaction_file_prefix)
                keypoints = alignment.keypoints

                published_on_str = new Date(reaction.release_date)
                published_on = Math.floor(published_on_str.getTime() / 1000)
                video_values = {
                  $vid: reaction_vid, 
                  $channel: reaction.reactor, 
                  $channel_id: reaction.channelId, 
                  $title: reaction.title, 
                  $description: reaction.description, 
                  $views: reaction.views, 
                  $duration: reaction.duration,
                  $published_on: published_on 
                }
                reaction_values = {
                  $vid: reaction_vid, 
                  $song_key: song,
                  $channel: reaction.reactor,
                  $keypoints: JSON.stringify(keypoints)
                }   
                         
                if !row?
                  console.log("INSERTING NEW REACTION VIDEO: #{reaction_values.$channel} for #{song}")
                  db.run \
                    """INSERT INTO Video (vid,  channel,  channel_id,  title,  description,  views,  duration, published_on) VALUES 
                                        ($vid, $channel, $channel_id, $title, $description, $views, $duration, $published_on)""", 
                    video_values


                  console.log("INSERTING NEW REACTION: #{reaction_values.$channel} for #{song}")
                  db.run \
                    "INSERT INTO Reaction (vid, song_key, channel, keypoints) VALUES ($vid, $song_key, $channel, $keypoints)", 
                    reaction_values
                
                else
                  console.log("UPDATING REACTION VIDEO #{reaction_values.$channel} for #{song}")
                  db.run \
                    """UPDATE Video SET channel=$channel, channel_id=$channel_id, title=$title, description=$description, views=$views, duration=$duration, published_on=$published_on
                                        WHERE vid=$vid""", video_values

                  console.log("UPDATING REACTION #{reaction_values.$channel} for #{song}")

                  reaction_values = {  # annoying, node-sqlite3 errors if there are unused values
                    $vid: reaction_values.$vid, 
                    $keypoints: reaction_values.$keypoints
                  }

                  db.run "UPDATE Reaction SET keypoints=$keypoints WHERE vid=$vid", reaction_values

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
bus.http.use('/assets', express.static('assets'))

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
    <script src="#{prefix}/client/shared.coffee"></script>
    <script src="#{prefix}/client/tooltips.coffee"></script>
    <script src="#{prefix}/client/channels.coffee"></script>
    <script src="#{prefix}/client/asides.coffee"></script>
    <script src="#{prefix}/client/inventory.coffee"></script>
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


      #tooltip .downward_arrow {
        width: 0; 
        height: 0; 
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;  
        border-top: 10px solid black;
      }
      #tooltip .upward_arrow {
        width: 0; 
        height: 0; 
        border-left: 10px solid transparent;
        border-right: 10px solid transparent;  
        border-bottom: 10px solid black;
      }

    </style>


    </head> 
    <body>

    </body>
    </html>
      """

  res.send(html)

