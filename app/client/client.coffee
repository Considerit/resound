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
  @local.included_in ?= ' '

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

    DIV null, 
      SPAN null,
        'included in: '

      SELECT 
        style: 
          fontSize: 18
        value: @local.included_in
        onChange: (e) => 
          @local.included_in = e.target.value
          save @local 

        for n,idx in mentioned_in_options
          OPTION 
            value: n
            style: {}
            n


    for channel_info in all

      if @local.filters[ channel_info.auto or ''  ]

        if (@local.mentioned_in == ' ' or @local.mentioned_in in channel_info.mentioned_in) and 
           (@local.included_in == ' ' or @local.included_in in channel_info.included_in)
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

  tasks = ['inventory', 'alignment', 'asides', 'reactors', 'backchannels', 'composition']

  loc = retrieve('location')

  parts = loc.path.split('/')

  song_config = retrieve("/song_config/#{song}")

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


      BUTTON
        onClick: =>
          editor = retrieve('json_editor')
          editor.active = !editor.active
          if editor.active
            editor.song = @props.song
          save editor
        'JSON'

    DIV
      style:
        backgroundColor: 'white'
        padding: '24px 18px'
        display: 'flex'

      if task == 'composition'
        COMPOSITION
          song: @props.song
      else if task == 'inventory'
        INVENTORY
          song: @props.song
      else
        ALIGNMENT
          song: @props.song   
          task: task   

      if retrieve('json_editor').active and song_config.config
        DIV
          onMouseEnter: (evt) ->
            document.body.style.overflow = 'hidden'
          onMouseLeave: (evt) ->
            document.body.style.overflow = ''

          style: 
            resize: 'horizontal'
            overflowY: 'auto'
            flexGrow: 1
            # flexShrink: 1
            padding: '0 10px'
            minWidth: 450
            position: 'sticky';
            top: 0
            alignSelf: 'flex-start'

          JSONEditorTextArea
            json: song_config.config
            key: "#{JSON.stringify(song_config.config).length}" # update text area if it changes elsewhere
            onChange: (txt) => 
              @local.updated_json = JSON.parse(txt)
              save @local
            onSave: =>
              if @local.updated_json
                song_config.config = @local.updated_json
                save song_config
                @local.updated_json = null
          # JSONEditorWithSchema()



dom.COMPOSITION = ->
  song = @props.song


  DIV null,

    VIDEO
      src: "/media/#{song}/#{song} (composition).mp4"
      controls: true
      playsinline: true
      width: 600


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
        if @local.channel_recommendation_filters[ channels[reaction.channelId]?.auto or '' ] && @local.included_filters[ reaction.download or false ]

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

  channel_warning = my_channel?.auto == 'exclude'

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


  downloaded_reactions.sort( (a,b) -> if a.reactor.toLowerCase().trim() < b.reactor.toLowerCase().trim() then -1 else 1     )


  # downloaded_reactions = [downloaded_reactions[0]]

  task = @props.task

  @registered_media ?= {}


  if @force_selection
    select_all = true
    @force_selection = false
  if @force_deselection
    deselect_all = true
    @force_deselection = false


  @local.page ?= 0
  @local.per_page = 10


  pagination = =>
    DIV null,

      if @local.page > 0
        BUTTON 
          onClick: => 
            @local.page -= 1
            if @local.page < 0
              @local.page = 0
            save @local
          'previous'

      BUTTON
        onClick: =>
          @local.page += 1
          save @local
        'next page'


  DIV null,


    DIV
      className: 'process-actions'
      style: 
        position: 'sticky'
        top: 0
        zIndex: 9
        display: 'flex'
        alignSelf: 'flex-start'
        backgroundColor: 'white'


      BUTTON 
        key: "play_all #{@local.play_all}"
        onClick: => 
          @local.play_all = !@local.play_all
          song_playing = false

          selected = @getDOMNode().querySelector('[data-is-selected=true]')

          for k,v of @registered_media
            is_song = v.is_song
            mute = is_song && song_playing
            is_audible = v.play(@local.play_all, mute, !selected)
            song_playing ||= is_song && is_audible

        if @local.play_all
          'Pause all'
        else
          'Play selected'

      BUTTON 
        key: "select_all #{@local.all_selected}"
        onClick: => 
          @local.all_selected = !@local.all_selected
          if @local.all_selected
            @force_selection = true
          else
            @force_deselection = true
          save @local

        if !@local.all_selected
          'Select all'
        else
          'Unselect all'

      BUTTON 
        key: 'hide_unselected'
        onClick: => 
          @local.hide_unselected = !@local.hide_unselected
          save @local
        
        if @local.hide_unselected
          'Show all'
        else
          'Hide unselected'


      if task == 'alignment'
        BUTTON 
          key: 'show reactions'
          onClick: => 
            @local.show_reactions = !@local.show_reactions
            save @local
          
          if @local.show_reactions
            'Hide reactions'
          else
            'Show reactions'

      if task == 'alignment'
        BUTTON 
          key: 'show paintings'
          onClick: => 
            @local.show_paintings = !@local.show_paintings
            save @local
          
          if @local.show_paintings
            'Hide paintings'
          else
            'Show paintings'

      pagination()

    UL 
      style:
        listStyle: 'none'
        paddingLeft: 24

      for reaction, i in downloaded_reactions
        metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
        if metadata.alignment and i >= @local.per_page * @local.page and i <= @local.per_page * (@local.page + 1)
          REACTION_ALIGNMENT
            song: song
            reaction: reaction
            task: task
            registered_media: @registered_media
            hide_unselected: @local.hide_unselected
            show_reaction: @local.show_reactions
            show_painting: @local.show_paintings
            force_selection: select_all
            force_deselection: deselect_all


dom.REACTION_ALIGNMENT = ->

  song = @props.song
  reaction = @props.reaction

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
  song_config = retrieve("/song_config/#{@props.song}")
  config = song_config.config

  reaction_file_prefix = reaction.file_prefix or reaction.reactor
  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  task = @props.task
  hide_unselected = @props.hide_unselected

  song_dir = ["/media", song].join('/')
  meta_dir = [song_dir, 'bounded'].join('/')
  reactions_dir = [song_dir, 'reactions'].join('/')


  aligned_video = [meta_dir, "#{reaction_file_prefix}-CROSS-EXPANDER" ].join('/')
  reaction_video = [reactions_dir, "#{reaction_file_prefix}" ].join('/')
  song_video = [song_dir, song].join('/')

  alignment_painting = [meta_dir, "#{reaction_file_prefix}-painting-3.png" ].join('/')

  isolated_backchannel = [meta_dir, "#{reaction_file_prefix}-isolated_backchannel.wav" ].join('/')

  if @props.force_selection
    @local.selected = true
  else if @props.force_deselection
    @local.selected = false

  if task == 'alignment'
    if @props.show_reaction
      vids = [song_video, aligned_video, reaction_video]      
    else
      vids = [song_video, aligned_video]
  else if task == 'backchannels'
    vids = [isolated_backchannel]    
  else if task == 'asides'
    vids = [song_video, reaction_video, reaction_video]
  else if task == 'reactors'
    vids = [aligned_video]
    for v in metadata.reactors or [] 
      vids.push [meta_dir, v ].join('/')
  else
    vids = [song_video, aligned_video, reaction_video]


  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  if hide_unselected and !@local.selected
    return LI null

  LI 
    key: "#{reaction.id} #{@local.key} #{task}"
    style: 
      backgroundColor: if @local.selected then '#dadada' else 'transparent'
      padding: '4px 12px'
      margin: '4px 0'
      borderRadius: 16
      display: if hide_unselected and !@local.selected then 'none' else 'flex'

    DIV 
      style: 
        display: 'flex'
        width: 250
        overflow: 'hidden'
        # height: 200

      BUTTON 
        title: 'Sync media for this reaction'
        style:
          backgroundColor: 'transparent'
          border: 'none'
          outline: 'none'
          cursor: 'pointer'
          opacity: if @local.disable_sync then .5 else 1

        onClick: => 
          @local.disable_sync = !@local.disable_sync
          save @local

        I 
          className: 'glyphicon glyphicon-link'


      if task in ['alignment', 'reactors', 'backchannels', 'asides']
        BUTTON
          style:
            cursor: 'pointer'
            border: 'none'
            backgroundColor: 'transparent'
            outline: 'none'
            margin: '0 4px'
          onClick: =>
            if task == 'alignment'
              if confirm('Reset will delete all processed metadata for this reaction. You sure?')
                action = 
                  key: "/action/#{reaction.id}"
                  action: 'delete'
                  scope: 'alignment'
                  reaction_id: reaction.id
                  song: song
                save action
            else if task == 'reactors'
              if confirm('Reset will delete all cropped reactor videos. You sure?')
                action = 
                  key: "/action/#{reaction.id}"
                  action: 'delete'
                  scope: 'cropped reactors'
                  reaction_id: reaction.id
                  song: song

                if confirm('Redo coarse face tracking too? Cancel means just redo fine-grained tracking.')
                  action.scope = 'cropped reactors including coarse'

                save action
            else if task == 'backchannels'
              if confirm('Reset will delete isolated backchannel files. You sure?')
                action = 
                  key: "/action/#{reaction.id}"
                  action: 'delete'
                  scope: 'isolated backchannel'
                  reaction_id: reaction.id
                  song: song
                save action
            else if task == 'asides'
              if confirm('Reset will delete all asides files for this reaction. You sure?')
                action = 
                  key: "/action/#{reaction.id}"
                  action: 'delete'
                  scope: 'asides'
                  reaction_id: reaction.id
                  song: song
                save action

          I
            className: 'glyphicon glyphicon-refresh'



      BUTTON
        onClick: => 
          navigator.clipboard.writeText(reaction.reactor)

        onDoubleClick: =>
          @local.selected = !@local.selected
          save @local


        style: 
          border: 'none'
          backgroundColor: 'transparent'
          outline: 'none'
          cursor: 'pointer'
          display: 'block'
          padding: "8px 10px"
          textAlign: 'left'

        reaction.reactor






    DIV 
      style: 
        display: 'flex'

      for video, idx in vids
        DIV
          key: "#{idx} #{reaction_video == video} #{song_video == video} #{aligned_video == video}"
          style: 
            marginRight: 12


          if task == 'backchannels'
            SYNCHRONIZED_AUDIO
              keep_synced: !@local.disable_sync
              selected: @local.selected
              audio: isolated_backchannel
              alignment_data: metadata.alignment.best_path
              is_reaction: false
              is_song: false
              registered_media: @props.registered_media   
              task: task  
              song: song   
              reaction_file_prefix: reaction_file_prefix       
          else
            SYNCHRONIZED_VIDEO
              keep_synced: !@local.disable_sync
              selected: @local.selected
              video: video
              alignment_data: metadata.alignment.best_path
              is_reaction: reaction_video == video
              is_song: song_video == video
              registered_media: @props.registered_media
              soundfile: if aligned_video == video then isolated_backchannel
              task: task
              song: song
              reaction_file_prefix: reaction_file_prefix
              onInitialize: do(idx) => (key) =>
                @local["video-#{idx}-time-key"] = key
              onTimeClicked: do(idx) => (e, time) => 
                @local["video-#{idx}-time"] = time
                save @local
              onVideoClicked: (e) => 
                if task == 'reactors'
                  x = event.offsetX
                  y = event.offsetY
                  w = e.target.clientWidth
                  h = e.target.clientHeight

                  @local.clicked_at = [x / w, y / h]
                  save @local

    if task == 'asides'

      insert_at = @local['video-0-time']
      start = @local['video-1-time']
      end = @local['video-2-time']

      DIV 
        style:
          paddingTop: 30

        if (config.asides?[reaction_file_prefix] or []).length > 0
          DIV null,
            LABEL null, 
              'Asides: '
            for aside, aside_idx in (config.asides?[reaction_file_prefix] or [])

              BUTTON 
                style: 
                  backgroundColor: if aside_idx == @local.editing_aside?[1] then '#ccc'
                onClick: do (aside, aside_idx) => => 
                  @local.editing_aside = [aside, aside_idx]
                  delete @local['video-0-time']
                  delete @local['video-1-time']
                  delete @local['video-2-time']
                  save @local

                "#{aside_idx}"

        if @local.editing_aside
          EDIT_ASIDE
            key: "aside-#{aside_idx}-#{@local.editing_aside}"
            song: @props.song
            aside: @local.editing_aside
            fresh: false
            time_state_key: @local.key
            registered_media: @props.registered_media 
            reaction_file_prefix: reaction_file_prefix
            on_cancel: =>
              delete @local.editing_aside
              delete @local['video-0-time']
              delete @local['video-1-time']
              delete @local['video-2-time']
              save @local
        
        else if insert_at? && start? && end?
          EDIT_ASIDE
            song: @props.song
            fresh: true
            time_state_key: @local.key
            registered_media: @props.registered_media
            reaction_file_prefix: reaction_file_prefix

    if task == 'alignment' and @props.show_painting
      A
        href: alignment_painting
        target: '_blank'
        style: 
            display: 'inline-block'

        IMG
          style: 
            height: 240
          src: alignment_painting



    if task == 'reactors' and (metadata.reactors or []).length > 0
      REACTOR_TASKS
        song: song
        reaction: reaction
        clicked_at: @local.clicked_at




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

  sort_asides = (asides) =>
    asides.sort (x,y) -> 
      diff = x[2] - y[2]
      if diff == 0
        return x[0] - y[0]
      return diff

    return asides

  split_aside = => 
    if !(end > @local.split_at > start)
      return # split position not between start and end

    asides = config.asides[reaction_file_prefix]

    split_one = aside.slice()
    split_two = aside.slice()

    split_one[1] = @local.split_at
    split_one[3] = false
    split_two[0] = @local.split_at


    asides.splice(aside_idx, 1)
    asides.push(split_one)
    asides.push(split_two)

    asides = sort_asides(asides)
    save song_config

    @local.split_at = null
    @local.show_split = false
    save @local


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
              for aside in asides
                [sstart, eend, iinsert_at, rrewind] = aside
                if (sstart == start && eend == end) ||   \
                   (iinsert_at == insert_at && eend == end) || \
                   (iinsert_at == insert_at && start == start)

                    aside[0] = start
                    aside[1] = end
                    aside[2] = insert_at
                    aside[3] = @local.rewind
                    save song_config
                    return

              new_aside = [start, end, insert_at, @local.rewind]
              asides.push(new_aside)
            else
              asides[aside_idx] = [start, end, insert_at, @local.rewind]

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
                asides = config.asides[reaction_file_prefix]
                asides.splice(aside_idx, 1)
                save song_config
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

        if @local.show_split
          DIV null, 

            INPUT
              type: 'text'
              onChange: (e) => 
                @local.split_at = parseFloat(e.target.value)
                save @local
            BUTTON
              onClick: (e) => 
                split_aside()
              'Split'

dom.REACTOR_TASKS = ->
  song = @props.song
  reaction = @props.reaction

  metadata = retrieve("/reaction_metadata/#{song}/#{reaction.id}")
  song_config = retrieve("/song_config/#{@props.song}")
  config = song_config.config

  reaction_file_prefix = reaction.file_prefix or reaction.reactor
  retrieve("/action/#{reaction.id}") # subscribe to actions on this reaction

  fname = metadata.reactors[0]
  parts = fname.split('-')
  vert = parts[parts.length - 1]
  hori = parts[parts.length - 2]

  override = config?.face_orientation?[reaction_file_prefix]

  sel_hori = override?[0] or hori
  sel_vert = override?[1] or vert

  num_reactors = config?.multiple_reactors?[reaction_file_prefix] or metadata.reactors.length

  featured = 'reaction_file_prefix' in (config?.featured or [])

  priority = config?.priority?[reaction_file_prefix] or (if featured then 75 else 50)


  DIV 
    style:
      fontSize: 16
      display: 'flex'
      flexDirection: 'column'

    DIV
      style: 
        display: 'flex'
      DIV null,

        SELECT 
          style: 
            fontSize: 18
          value: sel_hori
          onChange: (e) => 
            config.face_orientation ?= {}
            config.face_orientation[reaction_file_prefix] = [e.target.value, sel_vert]
            save song_config

          for val in ['left', 'right', 'center']
            OPTION 
              value: val
              style: {}
              val

      DIV null,
        SELECT  
          style: 
            fontSize: 18
          value: sel_vert
          onChange: (e) => 
            config.face_orientation ?= {}
            config.face_orientation[reaction_file_prefix] = [sel_hori, e.target.value]
            save song_config

          for val in ['up', 'down', 'middle']
            OPTION 
              value: val
              style: {}
              val
      
    DIV null,

      INPUT
        type: 'range'
        min: 0
        max: 100
        value: priority
        onChange: (e) =>
          config.priority ?= {} 
          config?.priority[reaction_file_prefix] = parseInt(e.target.value)
          save song_config


    if @props.clicked_at
      DIV null,
        SPAN null,
          "#{Math.round(@props.clicked_at[0] * 100)}% / #{Math.round(@props.clicked_at[1] * 100)}%"
        BUTTON 
          onClick: => 
            config.fake_reactor_position ?= {}
            config.fake_reactor_position[reaction_file_prefix] ?= []
            config.fake_reactor_position[reaction_file_prefix].push @props.clicked_at
            save song_config

          'Exclude as false positive reactor'

    SELECT
      style: 
        fontSize: 18
      value: num_reactors
      onChange: (e) =>
        config.multiple_reactors ?= {}
        config.multiple_reactors[reaction_file_prefix] = parseInt(e.target.value)
        save song_config

      for val in ['1','2','3','4','5']
        OPTION 
          value: val
          style: {}
          val


dom.SYNCHRONIZED_VIDEO = ->

  video = @props.video
  alignment_data = @props.alignment_data
  is_reaction = @props.is_reaction



  DIV null,

    VIDEO
      width: 320
      height: 240
      controls: true
      ref: 'video'
      'data-media': video
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


      onClick: (e) =>
        @props.onVideoClicked?(e)

    if is_reaction
      BEST_PATH_BAR
        alignment_data: alignment_data
        video: @props.video
    else
      DIV 
        style:
          height: 10
          width: '100%'

    if false and !is_reaction and !@props.is_song
      DIV
        ref: 'wavesurfer'
        style:
          height: 18
          width: 'calc(100% - 34px)'
          margin: '0 17px'



    TIME_DISPLAY
      time_state: "time-#{@local.key}"
      onTimeClicked: @props.onTimeClicked
      onInitialize: @props.onInitialize
      parent_key: @local.key


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
          video = document.querySelector("[data-media=\"#{@props.video}\"]")
          video.currentTime = segment.start



dom.BEST_PATH_BAR.refresh = ->
  if !@initialized
    video = document.querySelector("[data-media=\"#{@props.video}\"]")
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
      display: 'flex'
      alignItems: 'center'

    onClick: (ev) =>
      active = retrieve('active_number')
      active.number = time
      save active

      navigator.clipboard.writeText(active.number)

      @props.onTimeClicked?(ev, time)

    "#{time}"

dom.TIME_DISPLAY.up = ->
  @props.onInitialize?(@props.parent_key)





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
  delete @props.registered_media[@local.key]

dom.SYNCHRONIZED_VIDEO.refresh = ->

  vid = @refs.video.getDOMNode()
  is_reaction = @props.is_reaction
  alignment_data = @props.alignment_data



  if !@initialized
    @initialized = true


    @props.registered_media[@local.key] = 
      is_song: @props.is_song
      play: (play, mute) =>

        if !play
          vid.pause()
          vid.muted = false
        else if @props.selected
          vid.play()
          vid.muted = mute
          return mute
        return false 

      set_time: (base_time) => 
        @ignore = true
        vid.currentTime = base_time
        setTimeout =>
          @ignore = false
        , 1500

      synchronize: (base_time) =>
        if !@props.keep_synced
          return

        if is_reaction
          currentTime = get_base_time(vid.currentTime, alignment_data)

          if Math.abs(base_time - currentTime) > .00001
            translated_time = get_reaction_time(base_time, alignment_data)

            @ignore_seek = translated_time
            vid.currentTime = translated_time
        else
          if base_time != vid.currentTime
            @ignore_seek = base_time
            vid.currentTime = base_time

    if is_reaction
      vid.currentTime = get_reaction_time(0, alignment_data) # initialize to beginning of base video in the reaction

    handle_seek = (ev) =>
      if @ignore
        return

      ts = ev.target.currentTime
      if Math.abs(@ignore_seek - ts) < .00001
        return

      if is_reaction
        ts = get_base_time(ts, alignment_data)

      syncers = Object.keys(@props.registered_media)
      syncers.sort( (a,b) -> (if bus.cache[b]?.in_viewport then 1 else 0) - (if bus.cache[a]?.in_viewport then 1 else 0)  )
      for k in syncers 
        if k != @local.key
          @props.registered_media[k].synchronize(ts)

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









dom.SYNCHRONIZED_AUDIO = ->

  audio = @props.audio
  alignment_data = @props.alignment_data

  DIV   
    style: 
      display: 'flex'
      style:
        'align-items': 'center'

    DIV
      'data-media': audio
      'data-receive-viewport-visibility-updates': 2
      "data-component": @local.key
      "data-is-song": @props.is_song
      "data-is-reaction": @props.is_reaction
      "data-is-selected": @props.selected
      "data-keep-synced": @props.keep_synced

      ref: 'wavesurfer'
      style:
        height: 32
        width: 600
        # width: 'calc(100% - 34px)'
        margin: '0 17px'


    TIME_DISPLAY
      time_state: "time-#{@local.key}"


dom.SYNCHRONIZED_AUDIO.down = ->
  delete @props.registered_media[@local.key]

dom.SYNCHRONIZED_AUDIO.refresh = ->
  if !@refs.wavesurfer?
    return

  if !@initialized
    synchronize_wavesurfer = (base_time) =>

      if !@props.keep_synced
        return

      if base_time != wavesurfer.getCurrentTime
        @ignore_seek = base_time
        wavesurfer.setTime base_time

    wavesurfer = WaveSurfer.create
      container: @refs.wavesurfer.getDOMNode()
      waveColor: 'rgb(200, 0, 200)'
      progressColor: 'rgb(100, 0, 100)'
      url: encodeURI(@props.audio)
      height: 'auto'
      normalize: true


    # wavesurfer.on 'click', =>
    #   if !wavesurfer.isPlaying()
    #     wavesurfer.play()

    # wavesurfer.on 'dblclick', =>
    #   if wavesurfer.isPlaying()
    #     wavesurfer.pause()


    @initialized = true

    @props.registered_media[@local.key] = 
      is_song: @props.is_song
      play: (play, mute, force) =>
        if play && (force || @props.selected)
          wavesurfer.play()
          wavesurfer.setMuted(mute)
        else if !play
          wavesurfer.pause()
          wavesurfer.setMuted(false)
      synchronize: synchronize_wavesurfer

    handle_seek = (ts) =>

      if Math.abs(@ignore_seek - ts) < .00001
        return

      syncers = Object.keys(@props.registered_media)
      syncers.sort( (a,b) -> (if bus.cache[b]?.in_viewport then 1 else 0) - (if bus.cache[a]?.in_viewport then 1 else 0)  )
      for k in syncers 
        if k != @local.key
          @props.registered_media[k].synchronize(ts)

    wavesurfer.on 'seeking', handle_seek


    handle_time = (currentTime) =>
      time_state = retrieve("time-#{@local.key}")
      time_state.time = currentTime
      save time_state

    wavesurfer.on 'timeupdate', handle_time

    if @props.is_song
      wavesurfer.setVolume = .2
















dom.JSONEditorWithSchema = ->
  state = retrieve('json_editor')
  song_config = retrieve("/song_config/#{state.song}")

  DIV 
    ref: 'editor_holder'

dom.JSONEditorWithSchema.up = dom.JSONEditorWithSchema.refresh = ->
  element = @refs.editor_holder.getDOMNode()

  state = retrieve('json_editor')

  song_config = retrieve("/song_config/#{state.song}")

  if !song_config.config
    return


  if !@initialized

    editor = new JSONEditor element,
      theme: 'spectre'
      iconlib: 'spectre'
      schema:
        $ref: "/library/library_schema.json"
        format: "categories"
      startval: song_config.config
      ajax: true

    @initialized = true

# JSONEditorTextArea          
#   id: 'json'
#   key: md5(subdomain.customizations) # update text area if subdomain.customizations changes elsewhere
#   json: subdomain.customizations
#   onChange: (val) => 
#     @local.stringified_current_value = val



dom.JSONEditorTextArea = ->
  DIV
    style: 
      height: '90vh'
      position: 'relative'

    if @local.has_changes
      BUTTON
        style:
          postion: 'fixed'
          left: 0
          top: 0
        onClick: =>
          @props.onSave?()
          @local.has_changes = false
          save @local
        'Save'

    DIV 
      ref: 'json_editor'
      style: 
        height: '90vh'





dom.JSONEditorTextArea.up = dom.JSONEditorTextArea.refresh = ->

  if !@initialized
    editor = new JSONEditor @refs.json_editor.getDOMNode(), 
      mode: 'code'
      modes: ['tree', 'code']
      sortObjectKeys: true
      mainMenuBar: false

      onChange: =>
        try 
          @props.onChange? editor.getText()
          @props.onChangeJSON?()
          @local.has_changes = true
          save @local


        catch e 
          console.log 'Got error', e



    editor.set @props.json

    @initialized = true

