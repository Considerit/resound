window.extend = (obj) ->
  obj ||= {}
  for arg, idx in arguments 
    if idx > 0      
      for own name,s of arg
        if !obj[name]? || obj[name] != s
          obj[name] = s
  obj

dom.BODY = -> 
  
  loc = retrieve('location')

  if !loc.url or loc.url == ''
    loc.path = '/songs'
    loc.url = 'songs'
    save loc
    return DIV null

  DIV 
    style: 
      fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif'

    NAVBAR()

    if loc.url == 'songs' or loc.url == ''
      SONGS()

    else if loc.url == 'channels'
      CHANNELS()

    else if loc.url.startsWith('songs/')
      parts = loc.url.split('/')
      SONG
        song: decodeURI(parts[1])

dom.NAVBAR = ->
  loc = retrieve('location')

  current_path = '/' + loc.path.split('/')[1] 

  paths = [ ['/songs', 'Songs'], ['/channels', 'Channels'] ]

  asty = 
    color: 'black'
    textDecoration: 'none'
    fontWeight: 700
    display: 'inline-block'
    padding: '8px 24px'

  DIV 
    style:
      width: '100%'
      backgroundColor: '#f59568'

    for path in paths

      A
        style: extend {}, asty, 
          color: if path[0] == current_path then 'white' else 'black'
          backgroundColor: if path[0] == current_path then '#db5516' else '#f59568'

        href: path[0]
        path[1]



dom.CHANNELS = ->
  channels = retrieve('/channels').channels
  return DIV null if !channels

  all = Object.values(channels)

  @local.sort ?= 'activity'

  if @local.sort == 'inclusions'
    all.sort (b,a) ->
      ai = (a.included_in or []).length
      bi = (b.included_in or []).length

      if ai != bi
        ai - bi
      else
        ai = (a.mentioned_in or []).length
        bi = (b.mentioned_in or []).length
        ai - bi


  else
    all.sort( (b,a) -> (a.viewCount or 1) * (a.subscriberCount or 1) - (b.viewCount or 1) * (b.subscriberCount or 1)  )

  filters = ['include', 'exclude', 'eligible', '']
  if !@local.filters
    @local.filters ?= {}
    for filter in filters
      @local.filters[filter] = true


  songs = retrieve('/songs').songs

  return SPAN null if !songs

  mentioned_in_options = [' '].concat(songs.slice())

  @local.mentioned_in ?= ' '

  DIV null,

    DIV null, 
      SPAN null,
        'sort: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for sort in ['activity', 'inclusions']
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(sort) => => 
                @local.sort = sort
                save @local

              style: 
                backgroundColor: if @local.sort == sort then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              sort

    DIV null, 
      SPAN null,
        'channel recommendation: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for filter in filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.filters[filter] = !@local.filters[filter]
                save @local

              style: 
                backgroundColor: if @local.filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              if filter.length > 0
                filter
              else
                '<unset>'

    DIV null, 
      SPAN null,
        'mentioned in: '

      SELECT 
        style: 
          fontSize: 18
        value: @local.mentioned_in
        onChange: (e) => 
          @local.mentioned_in = e.target.value
          save @local 

        for n,idx in mentioned_in_options
          OPTION 
            value: n
            style: {}
            n

    for channel_info in all

      if @local.filters[ channel_info.auto or ''  ]

        if @local.mentioned_in == ' ' or @local.mentioned_in in channel_info.mentioned_in
          CHANNEL
            key: channel_info.channelId
            channel_info: channel_info


dom.CHANNEL = ->

  channel_info = @props.channel_info
  if channel_info.auto == 'exclude'
    color = '#ddb2b5'
  else if channel_info.auto == 'include'
    color = '#b9ddb2'
  else if channel_info.auto == 'eligible'
    color = '#e5e3f2'
  else
    color = '#fafafa'

  key = "/channel/#{channel_info.title}"
  retrieve(key)

  nfObject = new Intl.NumberFormat('en-US')

  DIV
    style:
      margin: '8px 12px'
      padding: '8px 12px'
      backgroundColor: color

    H4
      style:
        fontSize: 18


      A
        style: 
          color: 'black'

        href: "https://youtube.com/#{channel_info.customUrl}"
        target: '_blank'

        channel_info.title


    DIV null,
      "#{nfObject.format(channel_info.viewCount)} total views"
    DIV null,
      "#{nfObject.format(channel_info.subscriberCount)} subscribers"


    DIV null,

      SPAN null,
        "Included in"
        UL null,
          for song in channel_info.included_in or []
            LI  
              style:
                listStyle: 'none'
                padding: '0 6px'
              song

      SPAN null,
        "Mentioned in"
        UL null,
          for song in channel_info.mentioned_in or []
            LI  
              style:
                listStyle: 'none'
                padding: '0 6px'
              song

    UL null,
      for note, title of channel_info.notes or []
        title ?= '<none>'
        if title
          LI  
            style:
              listStyle: 'none'
              padding: '0 6px'

            A 
              target: "_blank"
              href: "https://www.youtube.com/channel/#{channel_info.channelId}/search?query=#{note}"
              "#{title}"


    DIV null,

      SELECT
        defaultValue: channel_info.auto
        style: 
          fontSize: 18
          display: 'block'
        onChange: (e) => 
          channel_info.auto = e.target.value
          to_save = 
              key: key
              val: channel_info
          save to_save

        for val in ['', 'eligible', 'include', 'exclude']
          OPTION
            key: val
            value: val
            val 


dom.SONGS = -> 
  all = retrieve('/songs').songs
  return DIV null if !all

  DIV null,

    UL null,
      for song in all
        LI null,

          A
            href: "/songs/#{song}"
            song



dom.SONG = -> 
  song = @props.song

  tasks = ['inventory', 'alignment', 'asides', 'reactors', 'backchannels']

  loc = retrieve('location')

  parts = loc.path.split('/')

  if parts.length < 4

    loc.path = loc.path + "/#{tasks[0]}"
    save loc
    return SPAN null

  else
    task = parts[3]

  asty = 
    color: 'black'
    textDecoration: 'none'
    padding: '8px 14px'
    fontWeight: 700
    display: 'inline-block'


  DIV null,

    DIV 
      style:
        width: '100%'
        backgroundColor: '#68c4f5'
        padding: '0px 0px'

      SPAN
        style: 
          padding: '0 24px'
        song

      for avail in tasks

        A
          style: extend {}, asty,
            backgroundColor: if avail == task then '#368fbe'
            color: if avail == task then 'white' else 'black'
          href: [parts[0], parts[1], parts[2], avail].join('/')
          avail



    DIV
      style:
        backgroundColor: 'white'
        padding: '24px 18px'

      if task != 'inventory'
        ALIGNMENT
          song: @props.song   
          task: task   
      else
        INVENTORY
          song: @props.song


dom.INVENTORY = ->
  song = @props.song

  manifest = retrieve("/manifest/#{song}")

  channels = retrieve('/channels').channels
  return DIV null if !channels

  if !manifest.manifest
    return SPAN null

  reactions = Object.values(manifest.manifest.reactions)

  @local.sort ?= 'views'

  if @local.sort == 'views'
    reactions.sort (a,b) -> b.views - a.views
  else # channel subscriptions

    reactions.sort( (b,a) -> channels[a.channelId].subscriberCount - channels[b.channelId].subscriberCount  )

  channel_recommendation_filters = ['include', 'exclude', 'eligible', '']
  if !@local.channel_recommendation_filters
    @local.channel_recommendation_filters ?= {}
    for filter in channel_recommendation_filters
      @local.channel_recommendation_filters[filter] = true

  included_filters = [true, false]
  if !@local.included_filters
    @local.included_filters ?= {}
    for filter in included_filters
      @local.included_filters[filter] = true

  DIV null,
    DIV null, 
      SPAN null,
        'sort: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for sort in ['views', 'subscribers']
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(sort) => => 
                @local.sort = sort
                save @local

              style: 
                backgroundColor: if @local.sort == sort then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              sort

    DIV null, 
      SPAN null,
        'channel recommendation: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for filter in channel_recommendation_filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.channel_recommendation_filters[filter] = !@local.channel_recommendation_filters[filter]
                save @local

              style: 
                backgroundColor: if @local.channel_recommendation_filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              if filter.length > 0
                filter
              else
                '<unset>'

    DIV null, 
      SPAN null,
        'included: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for filter in included_filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.included_filters[filter] = !@local.included_filters[filter]
                save @local

              style: 
                backgroundColor: if @local.included_filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              "#{filter}"


    UL 
      style:
        listStyle: 'none'

      for reaction in reactions
        if @local.channel_recommendation_filters[ channels[reaction.channelId].auto or '' ] && @local.included_filters[ reaction.download or false ]

          MANIFEST_ITEM
            song: song
            reaction: reaction

dom.MANIFEST_ITEM = ->
  reaction = @props.reaction
  song = @props.song

  key = "/reaction/#{reaction.id}"
  retrieve(key)

  explicit = !!reaction.explicit
  included = !!reaction.download

  if included
    color = if explicit then '#91f2ab' else '#b9ddb2'
  else
    color = if explicit then '#ddb2b5' else '#fafafa'

  @local.show_iframe = @local.show_iframe or @local.in_viewport

  channels = retrieve('/channels').channels
  my_channel = channels[reaction.channelId]

  channel_warning = my_channel.auto == 'exclude'

  LI 
    'data-receive-viewport-visibility-updates': 1
    "data-component": @local.key

    style: 
      backgroundColor: color
      padding: '18px'
      margin: 9
      borderRadius: 8


    if channel_warning
      DIV 
        style:
          backgroundColor: 'red'
          color: 'white'
          fontSize: 20
          display: 'inline-block'
        'CHANNEL EXCLUDED'

    DIV
      style: 
        fontSize: 14
      "#{reaction.views} views"

    DIV
      style: 
        fontSize: 14
      "#{reaction.release_date}"

    H3 null,
      reaction.reactor

    DIV
      style: 
        fontSize: 16
        marginBottom: 8
      dangerouslySetInnerHTML: __html: reaction.title

    DIV null,
      BUTTON
        onClick: ->
          reaction.explicit = true
          reaction.download = true
          save {
            key: key,
            song: song,
            reaction: reaction
          }

        style: 
          fontSize: 18
          backgroundColor: 'green'
          margin: 9
          borderRadius: 8
          borderColor: 'transparent'
          color: 'white'
          fontWeight: 'bold'
          padding: '8px 12px'          
        "Pass"

      BUTTON 
        onClick: ->
          reaction.explicit = true
          reaction.download = false
          save {
            key: key,
            song: song,
            reaction: reaction
          }    

        style: 
          fontSize: 18
          backgroundColor: 'red'
          margin: 9          
          borderRadius: 8
          borderColor: 'transparent'
          color: 'white'
          fontWeight: 'bold'
          padding: '8px 12px'
        "Fail"    
      
    if @local.show_iframe
      IFRAME
        width: "560" 
        height: "315" 
        src: "https://www.youtube.com/embed/#{reaction.id}" 
        frameborder: "0" 
        allow: "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
        allowfullscreen: true
    else
      DIV
        style:
          width: 560
          height: 315 



dom.ALIGNMENT = -> 
  song = @props.song
  manifest = retrieve("/manifest/#{song}")

  channels = retrieve('/channels').channels
  return DIV null if !channels

  if !manifest.manifest?.reactions
    return SPAN null

  all_reactions = Object.values(manifest.manifest.reactions)

  downloaded_reactions = []
  for o in all_reactions
    if o.download
      downloaded_reactions.push o


  # downloaded_reactions = [downloaded_reactions[0]]

  task = @props.task

  @registered_synchronized_video ?= {}

  DIV null,


    DIV
      'data-receive-viewport-visibility-updates': 1
      "data-component": @local.key
      style: 
        position: if !@local.in_viewport then 'fixed' else 'relative'
        top: 0
        zIndex: 9
        display: 'flex'

      if task == 'alignment'

        BUTTON 
          onClick: => 
            @local.play_all = !@local.play_all
            root = @getDOMNode()
            all_vids = Array.from(root.getElementsByTagName('video'))

            song_playing = false
            for v in all_vids
              if v.dataset.isSelected == 'true'
                if @local.play_all
                  v.play()
                  if song_playing
                    v.mute = true
                  if v.dataset.isSong == 'true'
                    song_playing = true
                else
                  v.pause()
                  v.muted = false
          if @local.play_all
            'Pause all'
          else
            'Play selected'

      BUTTON 
        onClick: => 
          @local.hide_unselected = !@local.hide_unselected
          save @local
        
        if @local.hide_unselected
          'Show all'
        else
          'Hide unselected'


      if task == 'asides'
        ASIDE_MAKER()

    
    UL 
      style:
        listStyle: 'none'

      for reaction in downloaded_reactions
        metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
        if metadata.alignment
          REACTION_ALIGNMENT
            song: song
            reaction: reaction
            task: task
            synchronization_registry: @registered_synchronized_video
            hide_unselected: @local.hide_unselected



dom.REACTION_ALIGNMENT = ->
  song = @props.song
  reaction = @props.reaction
  task = @props.task
  hide_unselected = @props.hide_unselected

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")

  song_dir = ["/media", song].join('/')
  meta_dir = [song_dir, 'bounded'].join('/')
  reactions_dir = [song_dir, 'reactions'].join('/')

  reaction_file_prefix = reaction.file_prefix or reaction.reactor

  aligned_video = [meta_dir, "#{reaction_file_prefix}-CROSS-EXPANDER" ].join('/')
  reaction_video = [reactions_dir, "#{reaction_file_prefix}" ].join('/')
  song_video = [song_dir, song].join('/')

  alignment_painting = [meta_dir, "#{reaction_file_prefix}-painting-3.png" ].join('/')

  isolated_backchannel = [meta_dir, "#{reaction_file_prefix}-isolated_backchannel.wav" ].join('/')

  if task == 'alignment' || task == 'backchannels'
    vids = [song_video, aligned_video]
  else if task == 'asides'
    vids = [song_video, reaction_video]
  else if task == 'reactors'
    vids = [reaction_video]
    for v in metadata.reactors or [] 
      vids.push [meta_dir, v ].join('/')
  else
    vids = [song_video, aligned_video, reaction_video]


  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction
  LI 
    key: "#{@local.key} #{task}"
    style: 
      backgroundColor: if @local.selected then '#dadada' else 'transparent'
      padding: '4px 12px'
      margin: '4px 0'
      borderRadius: 16
      display: if hide_unselected and !@local.selected then 'none' else 'flex'

    DIV 
      style: 
        display: 'inline-block'
        width: 130
        overflow: 'hidden'
        height: 200

      BUTTON
        onClick: =>
          @local.selected = !@local.selected
          save @local
        style: 
          border: 'none'
          backgroundColor: 'transparent'
          outline: 'none'
          cursor: 'pointer'
          display: 'block'
          padding: "40px 10px"

        reaction.reactor


      BUTTON 
        style:
          padding: "12px 18px"
          backgroundColor: 'transparent'
          border: '1px solid #ddd'
          outline: 'none'
          cursor: 'pointer'

        onClick: => 
          @local.disable_sync = !@local.disable_sync
          save @local

        if @local.disable_sync
          'Not syncing'
        else
          'Syncing'



    DIV 
      style: 
        display: 'flex'

      for video, idx in vids
        DIV
          key: "#{idx} #{reaction_video == video} #{song_video == video}"
          style: 
            marginRight: 12

          SYNCHRONIZED_VIDEO
            keep_synced: !@local.disable_sync
            selected: @local.selected
            playback_key: "#{song}/playback"
            video: video
            alignment_data: metadata.alignment.best_path
            is_reaction: reaction_video == video
            is_song: song_video == video
            synchronization_registry: @props.synchronization_registry
            soundfile: if aligned_video == video then isolated_backchannel

    if task == 'alignment'
      A
        href: alignment_painting
        target: '_blank'
        style: 
            display: 'inline-block'

        IMG
          style: 
            height: 240
          src: alignment_painting

    if task == 'alignment'
      BUTTON
        style:
          cursor: 'pointer'
        onClick: =>
          if confirm('Reset will delete all processed metadata for this reaction. You sure?')
            action = 
              key: "/action/#{reaction.id}"
              action: 'delete'
              scope: 'alignment'
              reaction_id: reaction.id
              song: song
            save action
        'reset'


    else if task == 'reactors'
      BUTTON
        style:
          cursor: 'pointer'
        onClick: =>
          if confirm('Reset will delete all cropped reactor videos. You sure?')
            action = 
              key: "/action/#{reaction.id}"
              action: 'delete'
              scope: 'cropped reactors'
              reaction_id: reaction.id
              song: song
            save action
        'reset'

    else if task == 'backchannels'
      BUTTON
        style:
          cursor: 'pointer'
        onClick: =>
          if confirm('Reset will delete isolated backchannel files. You sure?')
            action = 
              key: "/action/#{reaction.id}"
              action: 'delete'
              scope: 'isolated backchannel'
              reaction_id: reaction.id
              song: song
            save action
        'reset'

    else if task == 'asides'
      BUTTON
        style:
          cursor: 'pointer'
        onClick: =>
          if confirm('Reset will delete all asides files for this reaction. You sure?')
            action = 
              key: "/action/#{reaction.id}"
              action: 'delete'
              scope: 'asides'
              reaction_id: reaction.id
              song: song
            save action
        'reset'

dom.SYNCHRONIZED_VIDEO = ->

  video = @props.video
  alignment_data = @props.alignment_data
  is_reaction = @props.is_reaction
  playback_state = retrieve @props.playback_key
  if !playback_state.base_time?
    playback_state.base_time = 0
    save playback_state

  DIV null,

    VIDEO
      width: 320
      height: 240
      controls: true
      ref: 'video'
      'data-video': video
      'data-receive-viewport-visibility-updates': 2
      "data-component": @local.key
      "data-is-song": @props.is_song
      "data-is-reaction": @props.is_reaction
      "data-is-selected": @props.selected
      "data-keep-synced": @props.keep_synced

      SOURCE
        src: video + '.mp4'
        type: "video/mp4"

      SOURCE
        src: video + '.webm'
        type: "video/webm"

    if is_reaction
      BEST_PATH_BAR
        alignment_data: alignment_data
        video: @props.video
    else
      DIV 
        style:
          height: 10
          width: '100%'

    if !is_reaction and !@props.is_song
      DIV
        ref: 'wavesurfer'
        style:
          height: 18
          width: 'calc(100% - 34px)'
          margin: '0 17px'


    TIME_DISPLAY
      time_state: "time-#{@local.key}"


dom.BEST_PATH_BAR = ->
  alignment_data = @props.alignment_data
  video = @props.video

  duration = @local.duration or alignment_data[alignment_data.length - 1][1] / sample_rate
  segments = []
  
  last_reaction_end = 0

  for segment in alignment_data

    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate


    if last_reaction_end < reaction_start - 1
      segments.push 
        type: 'speaking'
        length: reaction_start - last_reaction_end
        start: last_reaction_end
        end: reaction_start


    segments.push 
      type: 'backchannel'
      length: reaction_end - reaction_start
      start: reaction_start
      end: reaction_end

    last_reaction_end = reaction_end

  if last_reaction_end < duration - 1
    segments.push 
      type: 'speaking'
      length: duration - last_reaction_end
      start: last_reaction_end
      end: duration

  DIV 
    style: 
      width: 'calc(100% - 34px)'
      margin: '0 17px'

    STYLE """
        .path_bar_segment {
          opacity: .7;
        }
        .path_bar_segment:hover {
          opacity: 1;
        }

      """

    for segment in segments
      BUTTON
        className: 'path_bar_segment'
        style:
          display: 'inline-block'
          height: 10
          cursor: 'pointer'
          width: "#{segment.length / duration * 100}%"
          backgroundColor: if segment.type == 'speaking' then '#ddb2b5' else '#ddd'
          outline: 'none'
          border: 'none'
          padding: 0
        onClick: do(segment) => (ev) =>
          video = document.querySelector("[data-video='#{@props.video}']")
          video.currentTime = segment.start



dom.BEST_PATH_BAR.refresh = ->
  if !@initialized
    video = document.querySelector("[data-video='#{@props.video}']")
    if video
      @initialized = true
      if video.duration and !isNaN(video.duration)
        @local.duration = video.duration
        save @local
      else
        video.addEventListener 'canplay', => 
          @local.duration = video.duration
          save @local


dom.TIME_DISPLAY = ->
  time = retrieve(@props.time_state).time or 0

  DIV
    style:
      color: 'black'
      cursor: 'pointer'

    onClick: (ev) =>
      active = retrieve('active_number')
      active.number = time
      save active

    "#{time}"





# Translates a time from reaction time to base time.
# alignment_data is a list of segment alignment tuples in the format 
#      (reaction_start, reaction_end, base_start, base_end, ...)
#   ...where we only need the first four values. 
#   reaction_start, reaction_end give the start and time of the segment
#   in reaction time. 
#   base_start, base_end give the start and end of the corresponding 
#   segment in the base time.
# Each of these values is an audio sample rate of 44100, so to get to 
# seconds, divide the value by 44100. 
# These values can be used to construct 
sample_rate = 44100
get_base_time = (reaction_vid_time, alignment_data) ->


  current_base = 0
  current_reaction = 0

  for segment in alignment_data
    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate

    seg_length = reaction_end - reaction_start


    if reaction_vid_time <= reaction_start + seg_length
      if reaction_vid_time >= reaction_start   # a part of the reaction in the base video
        btime = current_base + (reaction_vid_time - reaction_start)
        # console.log("FOUND BASE TIME #{btime} FROM #{reaction_vid_time}")
        return btime
      else # could not find a match, return the last base end
        return current_base

    else
      current_base = base_end
      current_reaction += seg_length
      last_reaction_end = reaction_end

  console.log("CANT FIND BASE TIME FROM #{reaction_vid_time}")
  return null



get_reaction_time = (base_time, alignment_data) ->

  if base_time == 0
    return alignment_data[0][0] / sample_rate

  current_base = 0

  for segment in alignment_data
    reaction_start = segment[0] / sample_rate
    reaction_end = segment[1] / sample_rate
    base_start = segment[2] / sample_rate
    base_end = segment[3] / sample_rate

    console.assert( Math.abs(  Math.abs(reaction_start - reaction_end) - Math.abs(base_start - base_end) ) < .0001    )
    seg_length = reaction_end - reaction_start

    if base_time <= current_base + seg_length
      rtime = reaction_start + (base_time - base_start)
      # console.log("FOUND REACTION TIME #{rtime} FROM #{base_time}")
      return rtime

    else
      current_base = base_end

  console.log("CANT FIND REACTION TIME FROM #{base_time}")

  return null



dom.SYNCHRONIZED_VIDEO.down = ->
  delete @props.synchronization_registry[@local.key]

dom.SYNCHRONIZED_VIDEO.refresh = ->

  # console.log("[#{@local.key}] SYNCHRONIZING VIDEO REFRESH")
  playback_state = retrieve @props.playback_key
  vid = @refs.video.getDOMNode()
  is_reaction = @props.is_reaction
  alignment_data = @props.alignment_data


  synchronize_vid = (base_time) =>
    if !@props.keep_synced
      return

    if is_reaction
      currentTime = get_base_time(vid.currentTime, alignment_data)
      # console.log('reaction time', playback_state.base_time, vid.currentTime, currentTime )

      if Math.abs(base_time - currentTime) > .00001
        # console.log("[#{@local.key}] REACTING TO SEEK CONVERTING", playback_state.base_time)
        translated_time = get_reaction_time(base_time, alignment_data)

        @ignore_seek = translated_time
        vid.currentTime = translated_time
    else
      if base_time != vid.currentTime
        @ignore_seek = base_time
        vid.currentTime = base_time



  if !@initialized
    @initialized = true

    @props.synchronization_registry[@local.key] = synchronize_vid

    if is_reaction
      vid.currentTime = get_reaction_time(0, alignment_data) # initialize to beginning of base video in the reaction

    handle_seek = (ev) =>

      ts = ev.target.currentTime
      if Math.abs(@ignore_seek - ts) < .00001
        return

      if is_reaction
        ts = get_base_time(ts, alignment_data)

      syncers = Object.keys(@props.synchronization_registry)
      syncers.sort( (a,b) -> (if bus.cache[b]?.in_viewport then 1 else 0) - (if bus.cache[a]?.in_viewport then 1 else 0)  )
      for k in syncers 
        if k != @local.key
          @props.synchronization_registry[k](ts)

    vid.addEventListener 'seeked', handle_seek


    handle_time = (ev) =>
      time_state = retrieve("time-#{@local.key}")
      time_state.time = ev.target.currentTime
      save time_state

    vid.addEventListener 'timeupdate', handle_time

    if @props.is_song
      vid.volume = .2


  if @refs.wavesurfer && @refs.wavesurfer && !@wavesurfer_added
    @wavesurfer_added = true
    wavesurfer = WaveSurfer.create
      container: @refs.wavesurfer.getDOMNode()
      waveColor: 'rgb(200, 0, 200)'
      progressColor: 'rgb(100, 0, 100)'
      url: encodeURI(@props.soundfile)
      height: 'auto'
      normalize: true
      # url: encodeURI("http://#{retrieve('location').host}#{@props.soundfile}")


    wavesurfer.on 'click', =>
      if !wavesurfer.isPlaying()
        wavesurfer.play()

    wavesurfer.on 'dblclick', =>
      if wavesurfer.isPlaying()
        wavesurfer.pause()






dom.ASIDE_MAKER = ->
  active = retrieve('active_number')
  aside = retrieve('active_aside')
  
  @last_active_number ?= active.number * active.number / active.number

  if @last_active_number != active.number && @last_selected
    @last_selected.value = active.number
    @last_selected = null
    @last_active_number = active.number


  DIV 
    style:
      display: 'flex'

    INPUT 
      style:
        width: 70
      ref: 'start'
      onClick: (ev) =>
        @last_selected = ev.target
      onChange: (ev) =>
        @local.start = ev.target.value
        save @local
      type: 'number'

    INPUT 
      style:
        width: 70
      ref: 'end'
      onClick: (ev) =>
        @last_selected = ev.target
      onChange: (ev) =>
        @local.end = ev.target.value
        save @local
      type: 'number'

    INPUT 
      style:
        width: 70    
      ref: 'insert'
      onClick: (ev) =>
        @last_selected = ev.target
      onChange: (ev) =>
        @local.insert = ev.target.value
        save @local
      type: 'number'

    INPUT 
      style:
        width: 70    
      onChange: (ev) =>
        @local.repeat = ev.target.value
        save @local
      type: 'text'


    DIV 
      style: 
        marginLeft: 10

      "[ #{@refs.start?.getDOMNode().value or ""}, #{@refs.end?.getDOMNode().value or ""}, #{@refs.insert?.getDOMNode().value or ""}#{if @local.repeat then ", #{@local.repeat}" else ''}]"

