

###########################
# Aside data manipulation #
###########################

window.sort_asides = (asides) ->
  asides.sort (x,y) -> 
    diff = x[2] - y[2]
    if diff == 0
      return x[0] - y[0]
    return diff

  return asides


window.get_all_asides = (song) -> 
  manifest = retrieve("/manifest/#{song}")
  channels = retrieve('/channels').channels
  song_config = retrieve("/song_config/#{song}")
  
  song_dir = ["/media", song].join('/')
  meta_dir = [song_dir, 'bounded'].join('/')
  reactions_dir = [song_dir, 'reactions'].join('/')

  config = song_config.config

  if !channels || !manifest.manifest?.reactions
    return SPAN null

  all_reactions = Object.values(manifest.manifest.reactions)

  all_asides = []
  for reaction in all_reactions or []
    reaction_file_prefix = reaction.file_prefix or reaction.reactor
    asides = config.asides[reaction_file_prefix] or []
    sort_asides(asides)
    for aside, idx in asides
      all_asides.push 
        idx: idx
        reaction_file_prefix: reaction_file_prefix
        reactor: reaction.reactor
        reaction: reaction
        aside: aside
        media:  [reactions_dir, "#{reaction_file_prefix}" ].join('/')

  all_asides.sort (a,b) -> 
    if a.aside[2] == b.aside[2]
      a.aside[0] - b.aside[0]
    else 
      a.aside[2] - b.aside[2]

  return all_asides

window.update_aside = (song, reaction_file_prefix, idx, start, end, insert_at, rewind) -> 
  song_config = retrieve("/song_config/#{song}")
  config = song_config.config
  asides = config.asides[reaction_file_prefix]
  asides[idx] = [start, end, insert_at, rewind]
  sort_asides(asides)
  save song_config


window.split_aside = (song, aside_idx, split_at, reaction_file_prefix) -> 

  song_config = retrieve("/song_config/#{song}")
  config = song_config.config

  asides = config.asides[reaction_file_prefix]
  aside = asides[aside_idx]

  start = aside[0]
  end = aside[1]
  if !(end > split_at > start)
    return # split position not between start and end


  split_one = aside.slice()
  split_two = aside.slice()

  split_one[1] = split_at
  split_one[3] = false
  split_two[0] = split_at


  asides.splice(aside_idx, 1)
  asides.push(split_one)
  asides.push(split_two)

  asides = sort_asides(asides)
  save song_config


window.delete_aside = (song, aside_idx, reaction_file_prefix) -> 

  song_config = retrieve("/song_config/#{song}")
  config = song_config.config

  config.asides[reaction_file_prefix].splice(aside_idx, 1)
  save song_config


##########################################
# EDIT ASIDE (add on to a REACTION TASK) #
##########################################

dom.EDIT_ASIDE = ->

  time_state = retrieve(@props.time_state_key)  

  song_config = retrieve("/song_config/#{@props.song}")
  config = song_config.config

  insert_at = time_state["video-0-time"]
  start = time_state["video-1-time"]
  end = time_state["video-2-time"]

  reaction_file_prefix = @props.reaction_file_prefix

  if @props.fresh
    @local.rewind ?= 3

  else
    [aside, aside_idx] = @props.aside
    start ?= aside[0]  
    end ?= aside[1]      
    insert_at ?= aside[2]
    if aside.length > 3
      @local.rewind = aside[3]
    else
      @local.rewind = 3


  DIV null,

    BUTTON 
      onClick: => 
        vid_key = time_state['video-0-time-key']
        @props.registered_media[vid_key].set_time(insert_at)

        vid_key = time_state['video-1-time-key']
        @props.registered_media[vid_key].set_time(start)

        vid_key = time_state['video-2-time-key']
        @props.registered_media[vid_key].set_time(end)


      '<-- apply'

    DIV 
      style: 
        paddingTop: 20
        display: 'flex'
        flexDirection: 'column'
      DIV null,
        LABEL 
          style: 
            width: 80
            display: 'inline-block' 
          "Insert at:"

        INPUT
          value: insert_at
          type: 'text'
          onChange: (e) =>
            time_state['video-0-time'] = parseFloat(e.target.value)
            save time_state

        BUTTON 
          onClick: => 
            vid_key = time_state['video-0-time-key']
            @props.registered_media[vid_key].set_time(insert_at)

          "Go"

      DIV null,
        LABEL
          style: 
            width: 80
            display: 'inline-block'             
          "Start:"

        INPUT
          value: start
          type: 'text'
          onChange: (e) =>
            time_state['video-1-time'] = parseFloat(e.target.value)
            save time_state

        BUTTON 
          onClick: =>
            vid_key = time_state['video-1-time-key']
            @props.registered_media[vid_key].set_time(start)
          "Go"

      DIV null,
        LABEL 
          style: 
            width: 80
            display: 'inline-block'             
          "End:"

        INPUT
          value: end
          type: 'text'
          onChange: (e) =>
            time_state['video-2-time'] = parseFloat(e.target.value)
            save time_state

        BUTTON 
          onClick: => 
            vid_key = time_state['video-2-time-key']
            @props.registered_media[vid_key].set_time(end)

          "Go"

      DIV null,
        LABEL 
          style: 
            width: 80
            display: 'inline-block'             

          'Rewind:'

        INPUT 
          type: 'text'
          defaultValue: @local.rewind
          onChange: (e) => 
            @local.rewind = parseInt(e.target.value)
            save @local

      DIV 
        style:
          display: 'flex'
          paddingTop: 12

        BUTTON
          style: 
            flexGrow: 1
          onClick: => 
            config.asides ?= {}
            config.asides[reaction_file_prefix] ?= []
            asides = config.asides[reaction_file_prefix]


            if @props.fresh
              # possibly overwrite existing aside...
              for aside, idx in asides
                [sstart, eend, iinsert_at, rrewind] = aside
                if (sstart == start && eend == end) ||   \
                   (iinsert_at == insert_at && eend == end) || \
                   (iinsert_at == insert_at && start == start)

                  update_aside(@props.song, reaction_file_prefix, idx, start, end, insert_at, @local.rewind)
                  return

              new_aside = [start, end, insert_at, @local.rewind]
              asides.push(new_aside)
              @props.on_save?(new_aside, insert_at)
            else
              update_aside(@props.song, reaction_file_prefix, idx, start, end, insert_at, @local.rewind)

            asides = sort_asides(asides)
            save song_config


          if @props.fresh then 'Add Aside' else 'Update Aside'

        if !@props.fresh
          BUTTON
            style: 
              flexGrow: 1
            onClick: => 
              @props.on_cancel?()
            "cancel"



        if !@props.fresh
          BUTTON 
            style: 
              flexGrow: 0
              backgroundColor: 'transparent'
              border: 'none'

            onClick: => 
              if confirm("Are you sure you want to delete this aside?")
                delete_aside(@props.song, aside_idx, reaction_file_prefix)
            I 
              className: "glyphicon glyphicon-trash"

        if !@props.fresh
          BUTTON 
            style:
              flexGrow: 0
              backgroundColor: 'transparent'
              border: 'none'
            onClick: => 
              @local.show_split = !@local.show_split
              save @local

            I
              className: "glyphicon glyphicon-scissors"

        if !@props.fresh && @local.show_split
          DIV null, 

            INPUT
              type: 'text'
              onChange: (e) => 
                @local.split_at = parseFloat(e.target.value)
                save @local
            BUTTON
              onClick: (e) => 
                split_aside(config, aside_idx, @local.split_at, reaction_file_prefix)
                @local.split_at = null
                @local.show_split = false
                save @local
              'Split'



##########################################
# ASIDE SUMMARY AND INSERTION EDITOR
##########################################

dom.ASIDE_SUMMARY_AND_INSERTION_EDITOR = ->
  song = @props.song
  all_asides = get_all_asides(song)

  total_aside_duration = 0
  for aside,idx in all_asides
    total_aside_duration += aside.aside[1] - aside.aside[0]

  @local.changes ?= {}

  DIV null, 
    LABEL 
      style: 
        backgroundColor: 'deepskyblue'
        color: 'white'
        fontSize: 24
        marginBottom: 18
      "#{all_asides.length} asides of #{Math.floor(total_aside_duration/60)} min #{Math.round(total_aside_duration%60)} sec total duration"
    DIV 
      style: 
        display: 'grid'
        alignItems: 'center'
        justifyContent: 'space-evenly'


      for aside,row in all_asides
        key = "#{aside.reaction_file_prefix}-#{aside.idx}-#{aside.aside[0]}-#{aside.aside[1]}"

        do (aside, row, key) =>         
          [
            DIV 
              key: "#{key}-reactor"
              style: 
                gridRow: row + 1
                gridColumn: 1
              aside.reactor


            DIV
              key: "#{key}-audio"

              style: 
                gridRow: row + 1
                gridColumn: 2


              AUDIO
                style: 
                  width: 100
                controls: true
                "data-id": "#{aside.reactor}-#{aside.aside[0]}-#{aside.aside[1]}"
                "data-start": aside.aside[0]
                "data-end": aside.aside[1]
                src: aside.media + '.wav'



            DIV 
              key: "#{key}-duration"            
              style: 
                gridRow: row + 1
                gridColumn: 3
              "#{Math.round(aside.aside[1] - aside.aside[0])}s"

            DIV 
              key: "#{key}-insertion"            
              style: 
                gridRow: row + 1
                gridColumn: 4
              INPUT
                type: 'number'
                defaultValue: aside.aside[2]
                onChange: (e) => 
                  @local.changes[key] ?= aside.aside.slice()
                  @local.changes[key][2] = parseFloat(e.target.value)
                  save @local

            DIV 
              key: "#{key}-rewind"            
              style: 
                gridRow: row + 1
                gridColumn: 5
              INPUT
                type: 'text'
                defaultValue: aside.aside[3]
                onChange: (e) => 
                  rewind = parseFloat(e.target.value)
                  if isNaN(e.target.value)
                    rewind = e.target.value
                  @local.changes[key] ?= aside.aside.slice()
                  @local.changes[key][3] = rewind
                  save @local

            if key of @local.changes
              DIV 
                key: "#{key}-save"
                style: 
                  gridRow: row + 1
                  gridColumn: 6

                BUTTON null,
                  onClick: => 
                    updated_aside = @local.changes[key]
                    update_aside(song, aside.reaction_file_prefix, aside.idx, updated_aside[0], updated_aside[1], updated_aside[2], updated_aside[3])
                    delete @local.changes[key]
                    save @local
                  I 
                    className: "glyphicon glyphicon-floppy-save"

            DIV 
              key: "#{key}-trash"            
              style: 
                gridRow: row + 1
                gridColumn: 7

              BUTTON 
                style: 
                  flexGrow: 0
                  backgroundColor: 'transparent'
                  border: 'none'

                onClick: => 
                  if confirm("Are you sure you want to delete this aside?")
                    delete_aside(song, aside.idx, aside.reaction_file_prefix)
                
                I 
                  className: "glyphicon glyphicon-trash"

          ]


dom.ASIDE_SUMMARY_AND_INSERTION_EDITOR.refresh = ->
  @initialized ?= {}

  audios = @getDOMNode().querySelectorAll('audio')
  for audio in audios
    continue if audio.dataset.id in @initialized

    audio.ontimeupdate = (ev) =>
      audio = ev.currentTarget
      if audio.currentTime >= audio.dataset.end
          audio.currentTime = audio.dataset.start
          audio.pause()

    audio.onplay = (ev) =>
      audio = ev.currentTarget
      audio.currentTime = audio.dataset.start
      audio.play()

    @initialized[audio.dataset.id] = true



#########################
# DETAILED ASIDE EDITOR #
#########################



dom.ASIDE_EDITOR_LIST = -> 

  song = @props.song
  all_asides = get_all_asides(song)

  DIV null,


    COMPOSITE_ASIDE_AND_BASE_VIDEO_PLAYER
      song: song

    ASIDE_SUMMARY_AND_INSERTION_EDITOR
      song: song
        
    UL 
      style:
        listStyle: 'none'
        paddingLeft: 24
        marginTop: 48



      for aside,idx in all_asides
        ASIDE_EDITOR_ITEM
          key: "#{aside.reaction_file_prefix}-#{idx}-#{aside.idx}-#{all_asides.length}"
          aside: aside
          song: song
          my_key: "#{aside.reaction_file_prefix}-#{idx}-#{aside.idx}-#{all_asides.length}"




dom.ASIDE_EDITOR_ITEM = ->
  @local.loop_in_region ?= true
  @width = Math.round(screen.width * .9)

  @local.changes ?= {}
  @my_key = @props.my_key
  song = @props.song
  aside = @props.aside


  @save_changes = =>
    if @my_key of @local.changes
      updated_aside = @local.changes[@my_key]
      update_aside(song, aside.reaction_file_prefix, aside.idx, updated_aside[0], updated_aside[1], updated_aside[2], updated_aside[3])
      delete @local.changes[@my_key]

  LI 
    key: @props.my_key
    style: 
      width: @width
      padding: '24px 0'
    'data-receive-viewport-visibility-updates': 1
    "data-component": @local.key

    DIV
      style:
        display: 'flex'
        alignItems: 'center'
        marginBottom: 20

      DIV 
        style: 
          marginRight: 50
          fontSize: 24
          backgroundColor: 'lightsalmon'
          color: 'white'
          fontWeight: 'bold'
          padding: '3px 10px'

        "#{aside.reaction_file_prefix} (##{aside.idx})"


      INPUT 
        type: 'checkbox'
        defaultChecked: @local.loop_in_region
        onClick: (e) =>
          @local.loop_in_region = e.target.checked
          save @local

      LABEL 
        style:
          padding: '0 8px'
          margin:0
        'Loop in region'


      LABEL 
        style:
          padding: '0 8px'
          margin:0
        'Zoom:'

      if @local.zoom?

        INPUT 

          type: 'range'
          defaultValue: @local.zoom
          min: 1
          max: 50
          style:
            width: 200

          onInput: (e) => 
            minPxPerSec = Number(e.target.value)
            @ws.zoom(minPxPerSec)

      if @my_key of @local.changes
        DIV 
          style: 
            padding: '0 8px'

          BUTTON null,
            onClick: @save_changes

            I 
              className: "glyphicon glyphicon-floppy-save"

      if @local.split_at?
        DIV 
          style: 
            padding: '0 8px'

          BUTTON null,
            onClick: =>
              if confirm("Split this aside here?")
                @save_changes()

                split_aside(song, aside.idx, @local.split_at, aside.reaction_file_prefix)

            I 
              className: "glyphicon glyphicon-scissors"





    DIV
      ref: 'wavesurfer'
      style:
        height: 140
        width: 'calc(100% - 34px)'
        margin: '0 17px'
        # min-width: 400px;
        flex: 1




dom.ASIDE_EDITOR_ITEM.refresh = ->
  return unless @refs.wavesurfer && @refs.wavesurfer && !@wavesurfer_added && @local.in_viewport

  url = encodeURI(@props.aside.media) + '.wav'
  aside = @props.aside.aside

  @wavesurfer_added = true

  @ws = ws = wavesurfer = WaveSurfer.create
    container: @refs.wavesurfer.getDOMNode()
    waveColor: 'rgb(200, 0, 200)'
    progressColor: 'rgb(100, 0, 100)'
    url: url
    height: 'auto'
    normalize: true

  wsRegions = ws.registerPlugin WaveSurfer.Regions.create()

  # Give regions a random color when they are created
  

  ws.on 'decode', =>

    # pixels / sec


    # Regions
    aside_region = wsRegions.addRegion 
      start: aside[0]
      end: aside[1]
      # content: 'Resize me'
      color: randomColor()
      drag: false
      resize: true
      # minLength: 1
      # maxLength: 10



    reaction_duration = ws.getDuration()
    aside_duration = aside[1] - aside[0]

    zoom = .65 * @width / aside_duration

    @local.zoom = zoom
    save @local

    ws.setTime(aside[0] + aside_duration / 2)
    @ws.zoom(zoom)


    setTimeout =>
      ws.setTime aside[0]
    , 1000
    # Markers (zero-length regions)
    marker_region = wsRegions.addRegion
      start: aside[0] + (aside[1] - aside[0]) / 2
      content: '/'
      color: randomColor()

    wsRegions.enableDragSelection
      color: 'rgba(255, 0, 0, 0.1)'

    wsRegions.on 'region-updated', (region) =>
      if region.id == marker_region.id
        @local.split_at = region.start

      else if region.id == aside_region.id

        @local.changes[@my_key] ?= aside.slice()
        @local.changes[@my_key][0] = region.start
        @local.changes[@my_key][1] = region.end

      save @local 

    @activeRegion = null
    wsRegions.on 'region-out', (region) => 
      if @activeRegion == region
        if @local.loop_in_region
          region.play()
        else
          @activeRegion = null

    wsRegions.on 'region-clicked', (region, e) =>
      return if wavesurfer.isPlaying()

      e.stopPropagation() # prevent triggering a click on the waveform

      @activeRegion = region
      region.play()
      region.setOptions({ color: randomColor() })

    # Reset the active region when the user clicks anywhere in the waveform
    ws.on 'interaction', =>
      @activeRegion = null


  wavesurfer.on 'click', =>
    if !wavesurfer.isPlaying()
      wavesurfer.play()

  wavesurfer.on 'dblclick', =>
    if wavesurfer.isPlaying()
      wavesurfer.pause()


########################################
# COMPOSITE ASIDE AND BASE VIDEO PLAYER
########################################



window.colors_per_reactor = {}


construct_composite_base_and_aside_segments = (song) ->
  
  song_dir = ["/media", song].join('/')

  all_asides = get_all_asides(song)
  song_video = [song_dir, song].join('/')

  segments = []
  current_base_playhead = 0
  duration = 0
  last_rewind = 0

  unique_reactors = {}
  for aside in all_asides
    unique_reactors[aside.reactor] = true

  reactor_colors = generateColorPalette(Object.values(unique_reactors).length)
  for reactor, idx in Object.keys(unique_reactors)
    if reactor not of colors_per_reactor
      colors_per_reactor[reactor] = reactor_colors[idx]

  for aside in all_asides
    [start, end, insertion_point, rewind] = aside.aside

    if insertion_point - current_base_playhead > 0
      segments.push 
        video: song_video
        start: current_base_playhead - last_rewind
        end: insertion_point
        is_base_video: true
        color: '#ccc'

      duration += insertion_point - current_base_playhead + last_rewind

      current_base_playhead = insertion_point

    # colors_per_reactor[aside.reactor] ?= randomColor()
    segments.push
      video: aside.media
      start: start
      end: end
      is_base_video: false
      color: colors_per_reactor[aside.reactor]
      aside: aside

    duration += end - start
    last_rewind = rewind

    # bug: we don't know how long the base video actually is. At the end, we'll try 
    #      to add segments at times for the base video that are longer than the base video itself

  return [segments, duration]





dom.COMPOSITE_ASIDE_AND_BASE_VIDEO_PLAYER = ->
  return SPAN null if !@props.song

  song = @props.song

  [segments, total_duration] = construct_composite_base_and_aside_segments(song)

  player_width = screen.width * .9
  DIV 
    style: 
      marginBottom: 80

    VIDEO
      width: player_width
      height: 480
      controls: true
      ref: 'video'

      SOURCE
        type: "video/mp4"

      SOURCE
        type: "video/webm"

    DIV
      style:
        height: 50
        width: player_width
        display: 'flex'
      onClick: (event) =>
        clickX = event.clientX # X position within the viewport
        elementX = event.currentTarget.getBoundingClientRect().left # X position of the div within the viewport
        relativeX = clickX - elementX # X position within the div
        clickPercent = relativeX / player_width
        seekTo = clickPercent * total_duration

        @player.seek(seekTo)


      for segment in segments
        segment_playing = @local.currentSegment?.video == segment.video && @local.currentSegment?.end == segment.end

        DIV
          "data-tooltip": if !segment.is_base_video then "#{segment.aside.reactor}"
          style:
            backgroundColor: segment.color
            height: '100%'
            width: "#{ 100 * (segment.end - segment.start) / total_duration}%"
            border: if segment_playing then "1px solid #{segment.color}" else "1px solid transparent"
            position: 'relative'

          if segment_playing
            DIV
              style: 
                backgroundColor: "rgba(0,0,0,.25)"
                width: 1
                height: '100%'
                position: 'absolute'
                left:"#{100 * @local.time_in_segment / (segment.end - segment.start)}%"





dom.COMPOSITE_ASIDE_AND_BASE_VIDEO_PLAYER.refresh = ->
  song = @props.song
  return if !song 
  [segments, total_duration] = construct_composite_base_and_aside_segments(song)

  key = "#{segments.length}-#{total_duration}"
  if @initialized != key

    if @player?
      @player.tearDown()

    @player = new CompositeVideoPlayer segments, @refs.video.getDOMNode(), (currentSegment, time_in_segment) =>
      @local.currentSegment = currentSegment
      @local.time_in_segment = time_in_segment
      save @local
      console.log(@local)

    @initialized = key






class CompositeVideoPlayer
  constructor: (segments, videoElement, onTimeUpdate) ->
    @segments = segments
    @currentSegmentIndex = 0

    # Calculate the total duration and map segments with their cumulative start time.
    cumulativeTime = 0

    @segments = segments



    @videoElement = videoElement
    @onTimeUpdate = onTimeUpdate



    @handleTimeUpdate = => 

      currentSegment = @segments[@currentSegmentIndex]
      if @videoElement.currentTime >= currentSegment.end || @videoElement.duration <= @videoElement.currentTime
        @transitionToNextSegment()
        currentSegment = @segments[@currentSegmentIndex]

      @onTimeUpdate?(currentSegment, @videoElement.currentTime - currentSegment.start)


    @videoElement.addEventListener 'timeupdate', @handleTimeUpdate



  transitionToNextSegment: ->
    if @currentSegmentIndex < @segments.length - 1
      @currentSegmentIndex += 1
      @loadCurrentSegment()

  loadCurrentSegment: ->
    currentSegment = @segments[@currentSegmentIndex]

    @updateSources(currentSegment)
    @videoElement.currentTime = currentSegment.start
    @videoElement.play()

  updateSources: (segment) ->
    sources = @videoElement.getElementsByTagName('source')
    
    sources[0].src = segment.video + '.mp4'
    sources[1].src = segment.video + '.webm'

    @videoElement.load()


  seek: (timeInSeconds) ->
    # Find the correct segment for the given time.
    cumulativeTime = 0

    for segment, idx in @segments
      segmentDuration = segment.end - segment.start
     
      if timeInSeconds <= cumulativeTime + segmentDuration
        @currentSegmentIndex = idx
        seekTime = timeInSeconds - cumulativeTime
        @updateSources(segment)
        @videoElement.currentTime = segment.start + seekTime

        @videoElement.play()
        break
      cumulativeTime += segmentDuration

  tearDown: ->
    return if !@videoElement
    @videoElement.pause()
    @videoElement.removeAttribute('src')
    @videoElement.load()
    @videoElement.removeEventListener('timeupdate', @handleTimeUpdate)
    @videoElement = null








